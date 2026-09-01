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

# ----------------- PEAK MORPHOLOGY AUDITING -----------------
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
            # Amplitude discrepancy (smaller y coordinate is higher excursion)
            dy = pred_y[p_pk] - target_y[t_pk]
            dx = p_pk - t_pk
            peak_matches.append({"dx": float(dx), "dy": float(dy)})
            
    if len(peak_matches) == 0:
        return {"mean_peak_dx": 99.0, "mean_peak_dy": 99.0}
        
    df_m = pd.DataFrame(peak_matches)
    mean_dx = np.mean(np.abs(df_m['dx']))
    mean_dy = np.mean(np.abs(df_m['dy']))
    return {"mean_peak_dx": float(mean_dx), "mean_peak_dy": float(mean_dy)}

# ----------------- PEAK-AWARE HYBRID CENTERLINE -----------------
def extract_hybrid_centerline(mask, roi_gray, mode="hybrid"):
    """
    Extracts 1D centerline from binary mask.
    Baseline regions: standard I^2 subpixel centroid.
    QRS/sharp peaks: Uses local-extremum subpixel correction (quadratic fitting).
    """
    h, w = mask.shape
    y_coords = np.zeros(w)
    
    # 1. First Pass: Compute standard I^2 centroid centerline
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) > 0:
            inverted_gray = 255.0 - col_gray[active_indices]
            weights = (inverted_gray ** 2) + 1e-5
            y_coords[x] = np.sum(active_indices * weights) / np.sum(weights)
        else:
            y_coords[x] = np.nan
            
    if mode == "baseline":
        return y_coords
        
    # 2. Second Pass: Apply Peak-Aware subpixel refinement to QRS extrema
    # We locate local extrema (peaks) in the first-pass centerline
    peaks = detect_peaks_in_trace(y_coords)
    
    for pk_idx in peaks:
        # Check slope / curvature to confirm it is a sharp QRS peak region
        # curvature ~ second derivative: y(x-1) - 2*y(x) + y(x+1)
        if pk_idx > 1 and pk_idx < w - 2:
            dy_left = y_coords[pk_idx] - y_coords[pk_idx - 1]
            dy_right = y_coords[pk_idx] - y_coords[pk_idx + 1]
            # In inverted coordinates, peaks point downwards (negative y-excursions)
            # R peaks are sharp minima, so dy_left and dy_right are negative (slope goes up to peak, then down)
            # We measure absolute steepness
            steepness = max(abs(dy_left), abs(dy_right))
            
            if steepness > 2.0: # High-slope threshold indicating a sharp peak excursion
                # 3. Fit local subpixel quadratic curve around the darkest pixel center in neighboring columns
                # We fit: y = ax^2 + bx + c for columns x-1, x, x+1
                xs = []
                ys = []
                for offset in [-1, 0, 1]:
                    curr_x = pk_idx + offset
                    col_gray = roi_gray[:, curr_x]
                    col_bin = mask[:, curr_x]
                    active = np.where(col_bin > 0)[0]
                    if len(active) > 0:
                        # Locate the absolute darkest pixel in the mask column (raw R-peak center)
                        darkest_y = active[np.argmin(col_gray[active])]
                        xs.append(curr_x)
                        ys.append(float(darkest_y))
                        
                if len(xs) == 3:
                    # Solve system using numpy polynomial fit
                    poly = np.polyfit(xs, ys, 2)
                    # Extremum of ax^2 + bx + c is at x = -b / (2a)
                    a, b, c = poly
                    if abs(a) > 1e-5:
                        subpixel_x = -b / (2.0 * a)
                        subpixel_y = a * (subpixel_x ** 2) + b * subpixel_x + c
                        # Constrain subpixel_y to stay within the local boundary bounds
                        lower_bound = min(ys) - 1.5
                        upper_bound = max(ys) + 1.5
                        if lower_bound <= subpixel_y <= upper_bound:
                            y_coords[pk_idx] = float(subpixel_y)
                            
    return y_coords

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 Peak-Aware Hybrid Centerline Experiments...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Hybrid_Centerline")
    
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
            
            target_y = []
            for col_idx in range(orig_w):
                coords = np.where(target_mask[:, col_idx] > 0)[0]
                target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
            target_y = np.array(target_y, dtype=float)
            
            # Predict
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            binary_orig = (prob_map > 0.40).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_orig, connectivity=8)
            clean_mask = np.zeros_like(binary_orig)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 10:
                    clean_mask[labels == i] = 255
            
            # Compare configurations
            for mode in ["baseline", "hybrid"]:
                pred_y = extract_hybrid_centerline(clean_mask, roi_gray, mode=mode)
                
                # Global MAE
                valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
                global_mae = np.mean(np.abs(pred_y[valid_mask] - target_y[valid_mask])) if np.sum(valid_mask) > 0 else 99.0
                
                peak_fidelity = evaluate_peak_morphology(pred_y, target_y)
                peak_fidelity.update({
                    "case": case, "lead": lead, "mode": mode,
                    "global_mae": float(global_mae)
                })
                results.append(peak_fidelity)
                
    df_res = pd.DataFrame(results)
    df_res.to_csv("stage5_output/hybrid_experiments_results.csv", index=False)
    
    print("\n--- Exp 3 & 6: Peak-Aware Hybrid vs Baseline I2 Centroid ---")
    print(df_res.groupby("mode")[["mean_peak_dx", "mean_peak_dy", "global_mae"]].mean().to_string())
    
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/hybrid_experiments_results.csv")
        
    print("\nHybrid centerline experiments complete.")

if __name__ == "__main__":
    main()
