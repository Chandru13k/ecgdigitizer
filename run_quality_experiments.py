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
from mlflow_config import setup_mlflow, log_dataset_lineage
import mlflow

# ----------------- CUSTOM FIDELITY METRICS -----------------
def calculate_fidelity_metrics(pred_y, target_y):
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
    
    try:
        r_val, _ = pearsonr(p_valid, t_valid)
        if np.isnan(r_val): r_val = 0.0
    except:
        r_val = 0.0
        
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
        
    recall_1px = np.mean(diffs <= 1.0)
    recall_2px = np.mean(diffs <= 2.0)
    recall_3px = np.mean(diffs <= 3.0)
    
    return {
        "mae": float(mae), "rmse": float(rmse), "p95": float(p95), "p99": float(p99), "max_dev": float(max_dev),
        "pearson_r": float(r_val), "diff1_corr": float(r_d1), "diff2_corr": float(r_d2),
        "recall_1px": float(recall_1px), "recall_2px": float(recall_2px), "recall_3px": float(recall_3px)
    }

# ----------------- SUBPIXEL WEIGHTING FUNCTIONS -----------------
def get_weights(inverted_gray, probs, mode):
    # Mode configurations
    # mode in: "I", "sqrt_I", "I2", "P_I", "P_I2", "P_sqrt_I"
    I = inverted_gray
    P = probs
    
    if mode == "I":
        return I
    elif mode == "sqrt_I":
        return np.sqrt(I)
    elif mode == "I2":
        return I ** 2
    elif mode == "P_I":
        return P * I
    elif mode == "P_I2":
        return P * (I ** 2)
    elif mode == "P_sqrt_I":
        return P * np.sqrt(I)
    else:
        return I

# ----------------- SUBPIXEL CENTERLINE EXTRACTION -----------------
def extract_centerline_sweep(prob_map, roi_grayscale, strategy_name, threshold=0.5, weight_mode="I"):
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
        inverted_gray = 255.0 - col_gray[active_indices]
        
        if strategy_name == "median":
            y_coords.append(np.median(active_indices))
        elif strategy_name == "centroid":
            y_coords.append(np.mean(active_indices))
        elif strategy_name == "grayscale_weighted":
            weights = get_weights(inverted_gray, probs, weight_mode) + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
        elif strategy_name == "probability_weighted":
            weights = probs + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
        elif strategy_name == "prob_grayscale_weighted":
            # Probability-weighted subpixel combo
            weights = (probs * get_weights(inverted_gray, probs, weight_mode)) + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
        else:
            y_coords.append(np.nan)
            
    return np.array(y_coords, dtype=float)

# ----------------- REGIONAL ERROR FORENSICS -----------------
def analyze_regional_errors(pred_y, target_y):
    pred_y = np.array(pred_y, dtype=float)
    target_y = np.array(target_y, dtype=float)
    
    valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
    if np.sum(valid_mask) < 5:
        return {"qrs_mae": 0.0, "baseline_mae": 0.0}
        
    # Segment signals based on derivatives
    # Regions with high derivative (excursions/peaks/edges) vs flat baseline regions
    dy = np.zeros_like(target_y)
    dy[1:] = np.abs(np.diff(target_y))
    
    # We set a threshold of 3 pixels per step to classify QRS/notches/transitions vs baseline
    qrs_mask = (dy > 3.0) & valid_mask
    baseline_mask = (dy <= 3.0) & valid_mask
    
    diffs = np.abs(pred_y - target_y)
    qrs_mae = np.mean(diffs[qrs_mask]) if np.sum(qrs_mask) > 0 else 0.0
    baseline_mae = np.mean(diffs[baseline_mask]) if np.sum(baseline_mask) > 0 else 0.0
    
    return {
        "qrs_mae": float(qrs_mae),
        "baseline_mae": float(baseline_mae)
    }

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 Quality Experiments Suite...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Quality_Experiments")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth not found. Please compile/train first.")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    
    sweep_results = []
    subpixel_results = []
    regional_results = []
    
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
            
            # --- Exp 1: Threshold x Centerline Sweep ---
            thresholds = [0.4, 0.5, 0.6, 0.7]
            strategies = ["median", "centroid", "grayscale_weighted", "probability_weighted", "prob_grayscale_weighted"]
            
            for t in thresholds:
                for strat in strategies:
                    pred_y = extract_centerline_sweep(prob_map, roi_gray, strat, threshold=t, weight_mode="I")
                    metrics = calculate_fidelity_metrics(pred_y, target_y)
                    metrics.update({"case": case, "lead": lead, "threshold": t, "strategy": strat})
                    sweep_results.append(metrics)
                    
            # --- Exp 2: Subpixel Weighting Ablation (at best threshold=0.6) ---
            weight_modes = ["I", "sqrt_I", "I2", "P_I", "P_I2", "P_sqrt_I"]
            for mode in weight_modes:
                pred_y = extract_centerline_sweep(prob_map, roi_gray, "grayscale_weighted", threshold=0.6, weight_mode=mode)
                metrics = calculate_fidelity_metrics(pred_y, target_y)
                metrics.update({"case": case, "lead": lead, "weight_mode": mode})
                subpixel_results.append(metrics)
                
            # --- Exp 3: Regional Forensics (under default best config: threshold=0.6, weight_mode="I2") ---
            pred_y = extract_centerline_sweep(prob_map, roi_gray, "grayscale_weighted", threshold=0.6, weight_mode="I2")
            forensics = analyze_regional_errors(pred_y, target_y)
            forensics.update({"case": case, "lead": lead})
            regional_results.append(forensics)
            
    # Save results to DataFrames
    df_sweep = pd.DataFrame(sweep_results)
    df_subpixel = pd.DataFrame(subpixel_results)
    df_regional = pd.DataFrame(regional_results)
    
    # Save CSV artifacts
    os.makedirs("stage5_output", exist_ok=True)
    df_sweep.to_csv("stage5_output/quality_sweep_results.csv", index=False)
    df_subpixel.to_csv("stage5_output/quality_subpixel_results.csv", index=False)
    df_regional.to_csv("stage5_output/quality_regional_results.csv", index=False)
    
    # Print Mean Summaries
    print("\n--- Exp 1: Threshold x Centerline Sweep Mean MAE Summary ---")
    print(df_sweep.groupby(["threshold", "strategy"])["mae"].mean().to_string())
    
    print("\n--- Exp 2: Subpixel Weight Mode Mean MAE Summary ---")
    print(df_subpixel.groupby(["weight_mode"])["mae"].mean().to_string())
    
    print("\n--- Exp 3: Regional Forensic Mean MAE Summary ---")
    print(df_regional[["qrs_mae", "baseline_mae"]].mean().to_string())
    
    # Generate Error Map Plot for case 'Copy of 19_male' Rhythm_II
    # Under best config (threshold=0.6, weight_mode="I2")
    try:
        case = 'Copy of 19_male'
        lead = 'Rhythm_II'
        stage4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
        rois_path = f"stage4_output/{case}/lead_rois.json"
        pseudo_dir = f"pseudo_labels/{case}"
        img = cv2.imread(stage4_ref_path, cv2.IMREAD_GRAYSCALE)
        with open(rois_path, 'r') as f:
            rois = json.load(f)
        data = rois[lead]
        roi_gray = img[data['y_start']:data['y_end'], data['x_start']:data['x_end']]
        
        target_mask = cv2.imread(os.path.join(pseudo_dir, f"{lead}_mask.png"), cv2.IMREAD_GRAYSCALE)
        target_y = []
        for col_idx in range(roi_gray.shape[1]):
            coords = np.where(target_mask[:, col_idx] > 0)[0]
            target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
        target_y = np.array(target_y, dtype=float)
        
        pad_h = (32 - (roi_gray.shape[0] % 32)) % 32
        pad_w = (32 - (roi_gray.shape[1] % 32)) % 32
        roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = model(tensor_img.to(device))
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
        prob_map = prob[0:roi_gray.shape[0], 0:roi_gray.shape[1]]
        
        pred_y = extract_centerline_sweep(prob_map, roi_gray, "grayscale_weighted", threshold=0.6, weight_mode="I2")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot(target_y, label="Stage 3 Reference", color="black", alpha=0.7)
        plt.plot(pred_y, label="Pred (Grayscale Weighted I²)", color="red", alpha=0.7)
        plt.title("Fidelity Trace Overlay")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(pred_y - target_y), color="purple", label="Absolute Error Δy(x)")
        plt.title("Local Error Forensic Map Δy(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("stage5_output/quality_error_map.png")
        plt.close()
    except Exception as e:
        print(f"Plotting error map failed: {e}")
        
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/quality_sweep_results.csv")
        mlflow.log_artifact("stage5_output/quality_subpixel_results.csv")
        mlflow.log_artifact("stage5_output/quality_regional_results.csv")
        if os.path.exists("stage5_output/quality_error_map.png"):
            mlflow.log_artifact("stage5_output/quality_error_map.png")
            
    print("\nQuality experiments completed and logged to MLflow successfully.")

if __name__ == "__main__":
    main()
