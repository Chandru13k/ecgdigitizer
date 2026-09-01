import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from mlflow_config import setup_mlflow
import mlflow

# ----------------- R-PEAK COMPARISON UTILITIES -----------------
def detect_peaks_in_trace(y_values):
    # R peaks are local minima (since y=0 is at the top of the image)
    # We use find_peaks on inverted signal (or -y)
    valid_indices = np.where(~np.isnan(y_values))[0]
    if len(valid_indices) < 10:
        return []
    
    y_clean = y_values[valid_indices]
    # Find peaks on -y with a minimum prominence of 5 pixels to reject noise ripples
    peaks, _ = find_peaks(-y_clean, prominence=5, distance=20)
    return valid_indices[peaks]

def evaluate_peak_fidelity(pred_y, target_y):
    """
    Find corresponding peaks and measure displacement.
    """
    pred_peaks = detect_peaks_in_trace(pred_y)
    target_peaks = detect_peaks_in_trace(target_y)
    
    peak_matches = []
    
    for t_pk in target_peaks:
        # Find closest predicted peak within ±15 px range
        close_pks = pred_peaks[np.abs(pred_peaks - t_pk) <= 15]
        if len(close_pks) > 0:
            p_pk = close_pks[np.argmin(np.abs(close_pks - t_pk))]
            # Recall coordinate y values (inverted coordinate system: y value itself is amplitude)
            dy = pred_y[p_pk] - target_y[t_pk]
            dx = p_pk - t_pk
            peak_matches.append({"dx": float(dx), "dy": float(dy)})
            
    if len(peak_matches) == 0:
        return {"mean_peak_dx_error": 99.0, "mean_peak_dy_error": 99.0, "matched_peaks": 0}
        
    df_m = pd.DataFrame(peak_matches)
    mean_dx = np.mean(np.abs(df_m['dx']))
    mean_dy = np.mean(np.abs(df_m['dy']))
    
    return {
        "mean_peak_dx_error": float(mean_dx),
        "mean_peak_dy_error": float(mean_dy),
        "matched_peaks": len(peak_matches)
    }

def get_weights(inverted_gray, mode):
    I = inverted_gray
    if mode == "I2":
        return I ** 2
    elif mode == "I4":
        return I ** 4
    else:
        return I

def extract_centerline_sweep(prob_map, roi_grayscale, threshold=0.4, weight_mode="I2"):
    h, w = prob_map.shape
    y_coords = []
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    for x in range(w):
        col_bin = binary_mask[:, x]
        col_gray = roi_grayscale[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) == 0:
            y_coords.append(np.nan)
            continue
            
        inverted_gray = 255.0 - col_gray[active_indices]
        weights = get_weights(inverted_gray, weight_mode) + 1e-5
        y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
            
    return np.array(y_coords, dtype=float)

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 R-Peak and QRS Fidelity Experiments...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Peak_Fidelity")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth missing!")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    results = []
    
    for case in cases:
        stage4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
        rois_path = f"stage4_output/{case}/lead_rois.json"
        pseudo_dir = f"pseudo_labels/{case}"
        
        if not os.path.exists(stage4_ref_path) or not os.path.exists(rois_path):
            continue
            
        img = cv2.imread(stage4_ref_path, cv2.IMREAD_GRAYSCALE)
        with open(rois_path, 'r') as f:
            rois = json.load(f)
            
        for lead, data in rois.items():
            x1, x2 = data['x_start'], data['x_end']
            y1, y2 = data['y_start'], data['y_end']
            
            roi_gray = img[y1:y2, x1:x2]
            orig_h, orig_w = roi_gray.shape
            
            target_mask_path = os.path.join(pseudo_dir, f"{lead}_mask.png")
            if not os.path.exists(target_mask_path):
                continue
            target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Ground truth centerline y
            target_y = []
            for col_idx in range(orig_w):
                coords = np.where(target_mask[:, col_idx] > 0)[0]
                target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
            target_y = np.array(target_y, dtype=float)
            
            # Model inference
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            # Compare subpixel modes (Exp 3: I2 vs I4)
            for mode in ["I2", "I4"]:
                pred_y = extract_centerline_sweep(prob_map, roi_gray, threshold=0.40, weight_mode=mode)
                fidelity = evaluate_peak_fidelity(pred_y, target_y)
                
                # Global MAE
                valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
                global_mae = np.mean(np.abs(pred_y[valid_mask] - target_y[valid_mask])) if np.sum(valid_mask) > 0 else 99.0
                
                fidelity.update({
                    "case": case, "lead": lead, "weight_mode": mode,
                    "global_mae": float(global_mae)
                })
                results.append(fidelity)
                
    df_res = pd.DataFrame(results)
    df_res.to_csv("stage5_output/rpeak_fidelity_audit.csv", index=False)
    
    # Summarize mean outcomes
    print("\n--- Exp 1 & 3: R-Peak Fidelity Audit (I2 vs I4 comparison) ---")
    print(df_res.groupby("weight_mode")[["mean_peak_dx_error", "mean_peak_dy_error", "global_mae"]].mean().to_string())
    
    # Exp 2: Mask-vs-Centerline Forensics worst-case report
    # Find lead-by-lead worst-case instances under I2 mode
    df_i2 = df_res[df_res["weight_mode"] == "I2"].sort_values("mean_peak_dy_error", ascending=False)
    print("\n--- Exp 2: Top Worst-Case R-Peak Leads (Under I2 configuration) ---")
    print(df_i2[["case", "lead", "mean_peak_dy_error", "global_mae"]].head(5).to_string())
    
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/rpeak_fidelity_audit.csv")
        
    print("\nR-peak fidelity audit completed successfully.")

if __name__ == "__main__":
    main()
