import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
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
        return {"mae": 99.0, "rmse": 99.0, "p95": 99.0, "p99": 99.0, "max_dev": 99.0, "recall_1px": 0.0, "recall_2px": 0.0}
        
    p_valid = pred_y[valid_mask]
    t_valid = target_y[valid_mask]
    
    diffs = np.abs(p_valid - t_valid)
    mae = np.mean(diffs)
    rmse = np.sqrt(np.mean(diffs**2))
    p95 = np.percentile(diffs, 95)
    p99 = np.percentile(diffs, 99)
    max_dev = np.max(diffs)
    
    recall_1px = np.mean(diffs <= 1.0)
    recall_2px = np.mean(diffs <= 2.0)
    
    return {
        "mae": float(mae), "rmse": float(rmse), "p95": float(p95), "p99": float(p99), "max_dev": float(max_dev),
        "recall_1px": float(recall_1px), "recall_2px": float(recall_2px)
    }

def get_weights(inverted_gray, mode):
    I = inverted_gray
    if mode == "I":
        return I
    elif mode == "sqrt_I":
        return np.sqrt(I)
    elif mode == "I2":
        return I ** 2
    elif mode == "I3":
        return I ** 3
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
    print("Initializing Stage 5 Targeted Refinement Suite...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Refinement_Suite")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth missing!")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    
    per_lead_audit = []
    weight_mode_sweep = []
    threshold_sweep = []
    
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
            
            # --- Exp 1: Per-lead Audit under frozen baseline (T=0.40, I2) ---
            pred_y_base = extract_centerline_sweep(prob_map, roi_gray, threshold=0.40, weight_mode="I2")
            metrics_base = calculate_fidelity_metrics(pred_y_base, target_y)
            metrics_base.update({"case": case, "lead": lead})
            per_lead_audit.append(metrics_base)
            
            # --- Exp 2: QRS weighting sweep with I3 support ---
            modes = ["I", "sqrt_I", "I2", "I3"]
            for mode in modes:
                pred_y = extract_centerline_sweep(prob_map, roi_gray, threshold=0.40, weight_mode=mode)
                metrics = calculate_fidelity_metrics(pred_y, target_y)
                metrics.update({"case": case, "lead": lead, "weight_mode": mode})
                weight_mode_sweep.append(metrics)
                
            # --- Exp 3: Adaptive/Local threshold sweep ---
            thresholds = [0.35, 0.40, 0.45, 0.50]
            for t in thresholds:
                pred_y = extract_centerline_sweep(prob_map, roi_gray, threshold=t, weight_mode="I2")
                metrics = calculate_fidelity_metrics(pred_y, target_y)
                metrics.update({"case": case, "lead": lead, "threshold": t})
                threshold_sweep.append(metrics)
                
    df_audit = pd.DataFrame(per_lead_audit)
    df_weights = pd.DataFrame(weight_mode_sweep)
    df_thresh = pd.DataFrame(threshold_sweep)
    
    # Save CSV files
    os.makedirs("stage5_output", exist_ok=True)
    df_audit.to_csv("stage5_output/refinement_lead_audit.csv", index=False)
    df_weights.to_csv("stage5_output/refinement_weight_sweep.csv", index=False)
    df_thresh.to_csv("stage5_output/refinement_threshold_sweep.csv", index=False)
    
    print("\n--- Exp 1: Per-Lead Forensic Audit ---")
    print(df_audit.to_string())
    
    print("\n--- Exp 2: QRS local weighting sweep (including I3) Mean MAE ---")
    print(df_weights.groupby("weight_mode")["mae"].mean().to_string())
    
    print("\n--- Exp 3: Threshold Sweeps Mean MAE ---")
    print(df_thresh.groupby("threshold")["mae"].mean().to_string())
    
    with mlflow.start_run():
        mlflow.log_artifact("stage5_output/refinement_lead_audit.csv")
        mlflow.log_artifact("stage5_output/refinement_weight_sweep.csv")
        mlflow.log_artifact("stage5_output/refinement_threshold_sweep.csv")
        
    print("\nTargeted Refinement Suite complete.")

if __name__ == "__main__":
    main()
