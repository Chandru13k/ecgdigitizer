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

# ----------------- R-PEAK DETECTOR -----------------
def detect_peaks_in_trace(y_values):
    valid_indices = np.where(~np.isnan(y_values))[0]
    if len(valid_indices) < 10:
        return []
    y_clean = y_values[valid_indices]
    peaks, _ = find_peaks(-y_clean, prominence=5, distance=20)
    return valid_indices[peaks]

# ----------------- MAIN RUNNER -----------------
def main():
    print("Initializing Stage 5 R-Peak Precision & Pixel Support Study...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Peak_Precision")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Error: stage5_best_model.pth missing!")
        return
    model.eval()
    
    cases = ['Copy of 19_male', 'Copy of 20_male']
    
    pixel_support_audit = []
    
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
            
            # Predict probability map
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(roi_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                output = model(tensor_img.to(device))
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
            prob_map = prob[0:orig_h, 0:orig_w]
            
            # Get binary mask
            binary_mask = (prob_map > 0.40).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            clean_mask = np.zeros_like(binary_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 10:
                    clean_mask[labels == i] = 255
            
            # Audit peak-support columns
            target_peaks = detect_peaks_in_trace(target_y)
            for pk_x in target_peaks:
                ref_y = target_y[pk_x]
                
                # Retrieve predicted binary mask columns around peak column pk_x
                col_mask = clean_mask[:, pk_x]
                active_y_indices = np.where(col_mask > 0)[0]
                
                if len(active_y_indices) > 0:
                    # Predicted mask vertical coverage boundaries
                    pred_min_y = np.min(active_y_indices) # Highest peak coordinate (remember y=0 is at top)
                    pred_max_y = np.max(active_y_indices)
                    
                    # R peaks are local minima (pointing up visually, i.e., lowest y coordinate values).
                    # Check if the reference R-peak y coordinate lies inside the predicted mask column vertical interval.
                    contains_peak = pred_min_y <= ref_y <= pred_max_y
                    
                    # Distance between true peak tip (ref_y) and predicted peak extremity (pred_min_y)
                    # positive means predicted mask failed to reach the height of the true peak.
                    tip_distance = float(pred_min_y - ref_y)
                else:
                    contains_peak = False
                    tip_distance = np.nan
                    
                pixel_support_audit.append({
                    "case": case, "lead": lead, "pk_x": int(pk_x),
                    "ref_y": float(ref_y), "contains_peak": bool(contains_peak),
                    "tip_distance_px": tip_distance
                })
                
    df_audit = pd.DataFrame(pixel_support_audit)
    output_path = "stage5_output/rpeak_pixel_support_audit.csv"
    df_audit.to_csv(output_path, index=False)
    
    # ----------------- REVEAL CONCLUSIONS -----------------
    total_peaks = len(df_audit)
    contained_peaks = df_audit["contains_peak"].sum()
    pct_contained = (contained_peaks / total_peaks) * 100 if total_peaks > 0 else 0.0
    
    mean_tip_dist = df_audit["tip_distance_px"].dropna().mean()
    max_tip_dist = df_audit["tip_distance_px"].dropna().max()
    
    print("\n--- Exp 1: R-Peak Pixel-Support Analysis Outcomes ---")
    print(f"Total Reference R-Peaks Audited: {total_peaks}")
    print(f"R-Peaks where reference coordinate is inside predicted mask bounds: {contained_peaks} ({pct_contained:.1f}%)")
    print(f"Mean vertical tip truncation distance: {mean_tip_dist:.3f} pixels")
    print(f"Max vertical tip truncation distance: {max_tip_dist:.3f} pixels")
    
    with mlflow.start_run():
        mlflow.log_artifact(output_path)
        mlflow.log_metric("pct_peaks_contained_in_mask", pct_contained)
        mlflow.log_metric("mean_peak_truncation_pixels", mean_tip_dist)
        
    print("\nPixel-support precision study complete.")

if __name__ == "__main__":
    main()
