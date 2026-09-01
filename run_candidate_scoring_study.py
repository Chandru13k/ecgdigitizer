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

# ----------------- R-PEAK DETECTION -----------------
def detect_peaks_in_trace(y_values):
    valid_indices = np.where(~np.isnan(y_values))[0]
    if len(valid_indices) < 10:
        return []
    y_clean = y_values[valid_indices]
    peaks, _ = find_peaks(-y_clean, prominence=5, distance=20)
    return valid_indices[peaks]

# ----------------- CANDIDATE-SCORING EXTRACTOR -----------------
def extract_candidate_scoring_centerline(mask, roi_gray, prob_map, mode="baseline"):
    """
    Extracts centerline.
    Baseline: Standard I^2 subpixel centroid.
    Scoring: Continuity-constrained candidate selection.
    """
    h, w = mask.shape
    y_baseline = np.zeros(w)
    
    # Compute baseline
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        active = np.where(col_bin > 0)[0]
        if len(active) > 0:
            inverted_gray = 255.0 - col_gray[active]
            weights = (inverted_gray ** 2) + 1e-5
            y_baseline[x] = np.sum(active * weights) / np.sum(weights)
        else:
            y_baseline[x] = np.nan
            
    if mode == "baseline":
        return y_baseline
        
    # candidate-scoring trajectory
    y_scoring = y_baseline.copy()
    peaks = detect_peaks_in_trace(y_baseline)
    
    for pk_x in peaks:
        # Define window around peak (±5 columns)
        x_start = max(0, pk_x - 5)
        x_end = min(w, pk_x + 6)
        
        for curr_x in range(x_start, x_end):
            col_bin = mask[:, curr_x]
            col_gray = roi_gray[:, curr_x]
            active = np.where(col_bin > 0)[0]
            
            if len(active) > 0:
                # We rank local candidates inside this column
                # Candidates: I2-centroid, darkest pixel, min-y (upper envelope), max-y (lower envelope), max-probability
                candidates = []
                
                # 1. Darkest pixel
                darkest_y = active[np.argmin(col_gray[active])]
                candidates.append(float(darkest_y))
                
                # 2. Max probability
                col_prob = prob_map[:, curr_x]
                max_prob_y = active[np.argmax(col_prob[active])]
                candidates.append(float(max_prob_y))
                
                # 3. Upper Envelope
                candidates.append(float(np.min(active)))
                
                # 4. Standard I2 centroid
                candidates.append(float(y_baseline[curr_x]))
                
                best_y = y_baseline[curr_x]
                best_score = -9999.0
                
                # Evaluate each candidate y using local scoring criteria:
                # - Darkness (inverted gray value)
                # - Probability
                # - Continuity constraint: distance to neighboring baseline points
                for cand_y in set(candidates):
                    y_idx = int(np.clip(cand_y, 0, h - 1))
                    darkness = 255.0 - col_gray[y_idx]
                    prob_val = col_prob[y_idx]
                    
                    # Neighbor continuity penalty (difference to neighbor baseline)
                    left_idx = max(0, curr_x - 1)
                    right_idx = min(w - 1, curr_x + 1)
                    continuity_diff = 0.5 * (abs(cand_y - y_baseline[left_idx]) + abs(cand_y - y_baseline[right_idx]))
                    
                    # Score function
                    score = (darkness / 25.5) + (prob_val * 5.0) - (continuity_diff * 0.8)
                    
                    if score > best_score:
                        best_score = score
                        best_y = cand_y
                        
                y_scoring[curr_x] = best_y
                
    return y_scoring

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 R-Peak Profile, Mask Thickness, and Candidate Scoring Experiments...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Candidate_Scoring")
    
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
    
    thickness_stats = []
    
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
            if lead not in ["V2", "V3", "aVF", "V4"]:
                continue
                
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
            
            # 1. Measure thickness profile across morphological regions
            # baseline vs QRS peaks
            peaks = detect_peaks_in_trace(target_y)
            for col in range(orig_w):
                active = np.where(clean_mask[:, col] > 0)[0]
                thickness = len(active) if len(active) > 0 else 0
                
                # Check if it's near an R-peak column (±3 px)
                is_rpeak = any(abs(col - pk) <= 3 for pk in peaks)
                thickness_stats.append({
                    "lead": lead, "is_rpeak": is_rpeak, "thickness": thickness
                })
                
            # 2. Extract and benchmark Trajectories
            for mode in ["baseline", "scoring"]:
                pred_y = extract_candidate_scoring_centerline(clean_mask, roi_gray, prob_map, mode=mode)
                
                # Metrics
                valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
                global_mae = np.mean(np.abs(pred_y[valid_mask] - target_y[valid_mask])) if np.sum(valid_mask) > 0 else 99.0
                
                results.append({
                    "case": case, "lead": lead, "mode": mode, "global_mae": float(global_mae)
                })
                
    df_res = pd.DataFrame(results)
    df_thick = pd.DataFrame(thickness_stats)
    
    # Print outcomes
    print("\n--- Exp 2: Mask Thickness Audit Outcomes ---")
    print(df_thick.groupby("is_rpeak")["thickness"].describe().to_string())
    
    print("\n--- Exp 3: Candidate Scoring Extractor vs Baseline I2 Centroid ---")
    print(df_res.groupby("mode")["global_mae"].mean().to_string())
    
    # Save files
    df_res.to_csv("stage5_output/candidate_scoring_results.csv", index=False)
    
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/candidate_scoring_results.csv")
        mlflow.log_metric("baseline_avg_mae", df_res[df_res["mode"]=="baseline"]["global_mae"].mean())
        mlflow.log_metric("scoring_avg_mae", df_res[df_res["mode"]=="scoring"]["global_mae"].mean())
        
    print("\nCandidate-scoring trajectory study complete.")

if __name__ == "__main__":
    main()
