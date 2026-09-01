import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from mlflow_config import setup_mlflow, log_dataset_lineage, log_standard_metrics
import mlflow

# ----------------- CUSTOM FIDELITY METRICS -----------------
def calculate_fidelity_metrics(pred_y, target_y):
    # Filter valid points (non-null/NaN)
    pred_y = np.array(pred_y, dtype=float)
    target_y = np.array(target_y, dtype=float)
    
    valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
    n_valid = np.sum(valid_mask)
    if n_valid < 5:
        return {
            "mae": 99.0, "rmse": 99.0, "p95": 99.0, "p99": 99.0, "max_dev": 99.0,
            "pearson_r": 0.0, "diff1_corr": 0.0, "diff2_corr": 0.0,
            "recall_1px": 0.0, "recall_2px": 0.0, "recall_3px": 0.0
        }
        
    p_valid = pred_y[valid_mask]
    t_valid = target_y[valid_mask]
    
    diffs = np.abs(p_valid - t_valid)
    mae = np.mean(diffs)
    rmse = np.sqrt(np.mean(diffs**2))
    p95 = np.percentile(diffs, 95)
    p99 = np.percentile(diffs, 99)
    max_dev = np.max(diffs)
    
    # Pearson Correlation
    try:
        r_val, _ = pearsonr(p_valid, t_valid)
        if np.isnan(r_val):
            r_val = 0.0
    except:
        r_val = 0.0
        
    # Derivatives Correlation
    # Compute 1st/2nd derivative correlation by aligning indices
    # We'll calculate differences on contiguous blocks or simple diff
    d_pred = np.diff(pred_y)
    d_target = np.diff(target_y)
    d_valid = ~np.isnan(d_pred) & ~np.isnan(d_target)
    try:
        r_d1, _ = pearsonr(d_pred[d_valid], d_target[d_valid])
        if np.isnan(r_d1): r_d1 = 0.0
    except:
        r_d1 = 0.0
        
    dd_pred = np.diff(d_pred)
    dd_target = np.diff(d_target)
    dd_valid = ~np.isnan(dd_pred) & ~np.isnan(dd_target)
    try:
        r_d2, _ = pearsonr(dd_pred[dd_valid], dd_target[dd_valid])
        if np.isnan(r_d2): r_d2 = 0.0
    except:
        r_d2 = 0.0
        
    # Pixel feature recall
    recall_1px = np.mean(diffs <= 1.0)
    recall_2px = np.mean(diffs <= 2.0)
    recall_3px = np.mean(diffs <= 3.0)
    
    return {
        "mae": float(mae), "rmse": float(rmse), "p95": float(p95), "p99": float(p99), "max_dev": float(max_dev),
        "pearson_r": float(r_val), "diff1_corr": float(r_d1), "diff2_corr": float(r_d2),
        "recall_1px": float(recall_1px), "recall_2px": float(recall_2px), "recall_3px": float(recall_3px)
    }

# ----------------- CENTERLINE STRATEGIES -----------------
def extract_centerline_strategy(prob_map, roi_grayscale, strategy_name, threshold=0.5):
    """
    Extracts 1D centerline from probability maps using different strategies.
    roi_grayscale: shape (H, W), grayscale raw cropped lead image
    prob_map: shape (H, W), values in [0, 1]
    """
    h, w = prob_map.shape
    y_coords = []
    
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    for x in range(w):
        col_prob = prob_map[:, x]
        col_bin = binary_mask[:, x]
        col_gray = roi_grayscale[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) == 0:
            y_coords.append(np.nan)
            continue
            
        probs = col_prob[active_indices]
        # Invert grayscale (ECG trace is dark on white background, so 255 - gray)
        inverted_gray = 255.0 - col_gray[active_indices]
        
        if strategy_name == "median":
            y_coords.append(np.median(active_indices))
            
        elif strategy_name == "centroid":
            # Pure spatial centroid
            y_coords.append(np.mean(active_indices))
            
        elif strategy_name == "grayscale_weighted":
            # Weighted by pixel darkness
            weights = inverted_gray + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
            
        elif strategy_name == "probability_weighted":
            # Weighted by U-Net probability
            weights = probs + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
            
        elif strategy_name == "prob_grayscale_weighted":
            # Joint weight
            weights = (probs * inverted_gray) + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
            
        elif strategy_name == "skeleton":
            # Simple skeleton center: middle of the binary bounds
            y_coords.append((np.max(active_indices) + np.min(active_indices)) / 2.0)
            
        else:
            y_coords.append(np.nan)
            
    return np.array(y_coords, dtype=float)

# ----------------- MAIN RUNNER FOR ABLATIONS -----------------
def run_ablations():
    print("Initializing Stage 5 Controlled Ablation Experiments...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Ablations")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth not found. Train the model first.")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    results = {}
    
    for case in cases:
        stage4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
        rois_path = f"stage4_output/{case}/lead_rois.json"
        pseudo_dir = f"pseudo_labels/{case}"
        
        if not os.path.exists(stage4_ref_path) or not os.path.exists(rois_path):
            continue
            
        img = cv2.imread(stage4_ref_path, cv2.IMREAD_GRAYSCALE)
        with open(rois_path, 'r') as f:
            rois = json.load(f)
            
        case_results = {}
        
        for lead, data in rois.items():
            x1, x2 = data['x_start'], data['x_end']
            y1, y2 = data['y_start'], data['y_end']
            
            roi_grayscale = img[y1:y2, x1:x2]
            orig_h, orig_w = roi_grayscale.shape
            
            # Load Stage 3 validation reference mask for this lead
            target_mask_path = os.path.join(pseudo_dir, f"{lead}_mask.png")
            if not os.path.exists(target_mask_path):
                continue
            target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract target centerline (from clean Stage 3 mask)
            target_y = []
            for col_idx in range(orig_w):
                coords = np.where(target_mask[:, col_idx] > 0)[0]
                target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
            target_y = np.array(target_y, dtype=float)
            
            # Model Predict
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_grayscale, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            lead_experiments = {}
            
            # --- Exp 1: Threshold Sweep ---
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            t_sweep = {}
            for t in thresholds:
                pred_y = extract_centerline_strategy(prob_map, roi_grayscale, "median", threshold=t)
                t_sweep[f"threshold_{t}"] = calculate_fidelity_metrics(pred_y, target_y)
            lead_experiments["exp1_threshold_sweep"] = t_sweep
            
            # --- Exp 2 & 3: Soft-mask and Centerline strategies ---
            strategies = [
                "median", "centroid", "grayscale_weighted", 
                "probability_weighted", "prob_grayscale_weighted", "skeleton"
            ]
            strat_sweep = {}
            for strat in strategies:
                pred_y = extract_centerline_strategy(prob_map, roi_grayscale, strat, threshold=0.5)
                strat_sweep[strat] = calculate_fidelity_metrics(pred_y, target_y)
            lead_experiments["exp2_3_centerline_strategies"] = strat_sweep
            
            case_results[lead] = lead_experiments
            
        results[case] = case_results
        
    # Aggregate and output results
    output_path = "stage5_output/ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Ablation experiments complete. Saved results to {output_path}")
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_artifact(output_path)
        # Log representative metrics for aVF / Rhythm II
        for case in cases:
            if case in results and "aVF" in results[case]:
                aVF_res = results[case]["aVF"]["exp2_3_centerline_strategies"]
                for strat, metrics in aVF_res.items():
                    mlflow.log_param(f"{case}_aVF_{strat}_mae", metrics["mae"])
                    mlflow.log_param(f"{case}_aVF_{strat}_r1px", metrics["recall_1px"])
                    
            if case in results and "Rhythm_II" in results[case]:
                r_res = results[case]["Rhythm_II"]["exp2_3_centerline_strategies"]
                for strat, metrics in r_res.items():
                    mlflow.log_param(f"{case}_Rhythm_II_{strat}_mae", metrics["mae"])
                    mlflow.log_param(f"{case}_Rhythm_II_{strat}_r1px", metrics["recall_1px"])

if __name__ == "__main__":
    run_ablations()
