import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from mlflow_config import setup_mlflow
import mlflow

# ----------------- CALIBRATION FACTORS (Typical standard ECG) -----------------
# 10 mm / mV => If 1 mm = 10 pixels, then 1 mV = 100 pixels (10 mm * 10 px/mm).
# We'll use: Y_SCALE = 100.0 px / mV (so error_mV = error_px / 100.0)
# 25 mm / s => If 1 mm = 10 pixels, then 1 s = 250 pixels.
# We'll use: X_SCALE = 250.0 px / s => 0.25 px / ms (so error_ms = error_px / 0.25)
Y_SCALE = 100.0
X_SCALE = 0.25

# ----------------- FIDELITY & ECG DOMAIN METRICS -----------------
def calculate_ecg_metrics(pred_y, target_y):
    pred_y = np.array(pred_y, dtype=float)
    target_y = np.array(target_y, dtype=float)
    
    valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
    n_valid = np.sum(valid_mask)
    if n_valid < 5:
        return {
            "pixel_mae": 99.0, "mv_mae": 0.99, "ms_mae": 99.0, "pearson_r": 0.0,
            "snr_db": 0.0, "recall_1px": 0.0, "recall_2px": 0.0
        }
        
    p_valid = pred_y[valid_mask]
    t_valid = target_y[valid_mask]
    
    diffs = np.abs(p_valid - t_valid)
    pixel_mae = np.mean(diffs)
    
    # Convert to physical domains
    mv_mae = pixel_mae / Y_SCALE
    ms_mae = pixel_mae / X_SCALE
    
    # Pearson r
    try:
        r_val, _ = pearsonr(p_valid, t_valid)
        if np.isnan(r_val): r_val = 0.0
    except:
        r_val = 0.0
        
    # SNR (dB)
    signal_power = np.mean(t_valid ** 2) + 1e-8
    noise_power = np.mean(diffs ** 2) + 1e-8
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    recall_1px = np.mean(diffs <= 1.0)
    recall_2px = np.mean(diffs <= 2.0)
    
    return {
        "pixel_mae": float(pixel_mae),
        "mv_mae": float(mv_mae),
        "ms_mae": float(ms_mae),
        "pearson_r": float(r_val),
        "snr_db": float(snr_db),
        "recall_1px": float(recall_1px),
        "recall_2px": float(recall_2px)
    }

def detect_peaks_in_trace(y_values):
    valid_indices = np.where(~np.isnan(y_values))[0]
    if len(valid_indices) < 10:
        return []
    y_clean = y_values[valid_indices]
    peaks, _ = find_peaks(-y_clean, prominence=5, distance=20)
    return valid_indices[peaks]

def evaluate_peak_morphology(pred_y, target_y):
    pred_peaks = detect_peaks_in_trace(pred_y)
    target_peaks = detect_peaks_in_trace(target_y)
    
    peak_matches = []
    for t_pk in target_peaks:
        close_pks = pred_peaks[np.abs(pred_peaks - t_pk) <= 15]
        if len(close_pks) > 0:
            p_pk = close_pks[np.argmin(np.abs(close_pks - t_pk))]
            # Coordinate system matches: smaller coordinate values are higher excursions
            dy = pred_y[p_pk] - target_y[t_pk]
            dx = p_pk - t_pk
            peak_matches.append({"dx_ms": float(dx / X_SCALE), "dy_mv": float(dy / Y_SCALE)})
            
    if len(peak_matches) == 0:
        return {"r_peak_timing_error_ms": 99.0, "r_peak_amplitude_error_mv": 0.99}
        
    df_m = pd.DataFrame(peak_matches)
    mean_dx = np.mean(np.abs(df_m['dx_ms']))
    mean_dy = np.mean(np.abs(df_m['dy_mv']))
    
    return {
        "r_peak_timing_error_ms": float(mean_dx),
        "r_peak_amplitude_error_mv": float(mean_dy)
    }

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
        weights = (inverted_gray ** 2) + 1e-5
        y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
            
    return np.array(y_coords, dtype=float)

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 Final Generalization Validation (mV & ms domain analysis)...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Final_Validation")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth missing!")
        return
    model.eval()
    
    # We will run blind validation across the active cases in the workspace directory
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
            
            # Ground truth y
            target_y = []
            for col_idx in range(orig_w):
                coords = np.where(target_mask[:, col_idx] > 0)[0]
                target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
            target_y = np.array(target_y, dtype=float)
            
            # Pad & Predict
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            # Inference under frozen settings (T=0.40, I2 weights)
            pred_y = extract_centerline_sweep(prob_map, roi_gray, threshold=0.40, weight_mode="I2")
            
            # Calculate metrics
            ecg_res = calculate_ecg_metrics(pred_y, target_y)
            peak_res = evaluate_peak_morphology(pred_y, target_y)
            
            ecg_res.update(peak_res)
            ecg_res.update({"case": case, "lead": lead})
            results.append(ecg_res)
            
    df_final = pd.DataFrame(results)
    output_path = "stage5_output/final_validation_results.csv"
    df_final.to_csv(output_path, index=False)
    
    # Compile ranked summary reports
    print("\n--- Final Validation Metrics (Lead-by-Lead Means) ---")
    print(df_final.groupby("lead")[["pixel_mae", "mv_mae", "ms_mae", "r_peak_amplitude_error_mv", "r_peak_timing_error_ms", "snr_db"]].mean().to_string())
    
    print("\n--- Final Validation Metrics (Overall Generalization Mean) ---")
    print(df_final[["pixel_mae", "mv_mae", "ms_mae", "r_peak_amplitude_error_mv", "r_peak_timing_error_ms", "snr_db", "recall_1px", "recall_2px"]].mean().to_string())
    
    with mlflow.start_run():
        mlflow.log_artifact(output_path)
        # Log global averages
        mlflow.log_metric("global_mv_mae", df_final["mv_mae"].mean())
        mlflow.log_metric("global_ms_mae", df_final["ms_mae"].mean())
        mlflow.log_metric("global_rpeak_dy_mv", df_final["r_peak_amplitude_error_mv"].mean())
        mlflow.log_metric("global_rpeak_dx_ms", df_final["r_peak_timing_error_ms"].mean())
        
    print(f"\nFinal blind validation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
