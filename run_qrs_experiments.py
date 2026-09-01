import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from mlflow_config import setup_mlflow
import mlflow

# ----------------- CUSTOM FIDELITY METRICS -----------------
def calculate_fidelity_metrics(pred_y, target_y):
    pred_y = np.array(pred_y, dtype=float)
    target_y = np.array(target_y, dtype=float)
    
    valid_mask = ~np.isnan(pred_y) & ~np.isnan(target_y)
    n_valid = np.sum(valid_mask)
    if n_valid < 5:
        return {"mae": 99.0, "rmse": 99.0, "recall_1px": 0.0, "recall_2px": 0.0}
        
    p_valid = pred_y[valid_mask]
    t_valid = target_y[valid_mask]
    
    diffs = np.abs(p_valid - t_valid)
    mae = np.mean(diffs)
    rmse = np.sqrt(np.mean(diffs**2))
    recall_1px = np.mean(diffs <= 1.0)
    recall_2px = np.mean(diffs <= 2.0)
    
    return {
        "mae": float(mae), "rmse": float(rmse),
        "recall_1px": float(recall_1px), "recall_2px": float(recall_2px)
    }

def get_weights(inverted_gray, probs, mode):
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

def extract_centerline_sweep(prob_map, roi_grayscale, strategy_name, threshold=0.4, weight_mode="I"):
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
            weights = (probs * get_weights(inverted_gray, probs, weight_mode)) + 1e-5
            y_coords.append(np.sum(active_indices * weights) / np.sum(weights))
        else:
            y_coords.append(np.nan)
            
    return np.array(y_coords, dtype=float)

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 QRS Experiments...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_QRS_Experiments")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth missing!")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    
    t040_results = []
    regional_forensics = []
    mask_vs_extractor = []
    
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
            
            # Ground truth trace y
            target_y = []
            for col_idx in range(orig_w):
                coords = np.where(target_mask[:, col_idx] > 0)[0]
                target_y.append(np.median(coords) if len(coords) > 0 else np.nan)
            target_y = np.array(target_y, dtype=float)
            
            # Predict probability map
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            # --- Exp 1: T=0.40 subpixel weighting sweep ---
            weight_modes = ["I", "sqrt_I", "I2", "P_I", "P_I2", "P_sqrt_I"]
            for mode in weight_modes:
                pred_y = extract_centerline_sweep(prob_map, roi_gray, "grayscale_weighted", threshold=0.40, weight_mode=mode)
                metrics = calculate_fidelity_metrics(pred_y, target_y)
                metrics.update({"case": case, "lead": lead, "weight_mode": mode})
                t040_results.append(metrics)
                
            # --- Exp 2: QRS Amplitude & Width forensics ---
            dy = np.zeros_like(target_y)
            dy[1:] = np.abs(np.diff(target_y))
            qrs_mask = (dy > 3.0) & (~np.isnan(target_y))
            
            pred_y_best = extract_centerline_sweep(prob_map, roi_gray, "grayscale_weighted", threshold=0.40, weight_mode="I2")
            diffs = np.abs(pred_y_best - target_y)
            
            qrs_mae = np.mean(diffs[qrs_mask]) if np.sum(qrs_mask) > 0 else 0.0
            baseline_mae = np.mean(diffs[~qrs_mask & (~np.isnan(target_y))]) if np.sum(~qrs_mask & (~np.isnan(target_y))) > 0 else 0.0
            
            regional_forensics.append({
                "case": case, "lead": lead, 
                "qrs_mae": qrs_mae, "baseline_mae": baseline_mae
            })
            
            # --- Exp 3: Mask-vs-Extractor separation ---
            # Compare predicted binary mask overlap with Stage 3 foreground (before centerline)
            pred_bin = (prob_map > 0.40).astype(np.uint8) * 255
            
            # Segmentation Mask Error (Dice / IoU)
            intersection = np.sum((pred_bin > 0) & (target_mask > 0))
            union = np.sum((pred_bin > 0) | (target_mask > 0))
            mask_iou = intersection / (union + 1e-8)
            
            # Extractor Error (Difference between centroid and grayscale_weighted on the same mask)
            pred_y_centroid = extract_centerline_sweep(prob_map, roi_gray, "centroid", threshold=0.40)
            extractor_delta = np.nanmean(np.abs(pred_y_best - pred_y_centroid))
            
            mask_vs_extractor.append({
                "case": case, "lead": lead, 
                "mask_iou": mask_iou, "extractor_delta": extractor_delta
            })
            
    df_t040 = pd.DataFrame(t040_results)
    df_forensics = pd.DataFrame(regional_forensics)
    df_separation = pd.DataFrame(mask_vs_extractor)
    
    # Print summaries
    print("\n--- Exp 1: T=0.40 Subpixel Weight Sweeps Mean MAE ---")
    print(df_t040.groupby("weight_mode")["mae"].mean().to_string())
    
    print("\n--- Exp 2: QRS vs Baseline Regional Forensics Mean MAE ---")
    print(df_forensics[["qrs_mae", "baseline_mae"]].mean().to_string())
    
    print("\n--- Exp 3: Mask IoU vs Centerline Extractor Delta ---")
    print(df_separation[["mask_iou", "extractor_delta"]].mean().to_string())
    
    # Save CSVs
    df_t040.to_csv("stage5_output/qrs_sweep_results.csv", index=False)
    df_forensics.to_csv("stage5_output/qrs_forensics_results.csv", index=False)
    df_separation.to_csv("stage5_output/qrs_separation_results.csv", index=False)
    
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/qrs_sweep_results.csv")
        mlflow.log_artifact("stage5_output/qrs_forensics_results.csv")
        mlflow.log_artifact("stage5_output/qrs_separation_results.csv")
        
    print("\nQRS Experiments successfully complete.")

if __name__ == "__main__":
    main()
