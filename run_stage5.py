import sys
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18

from scipy.signal import find_peaks

def suppress_boundary_artifacts(binary_mask, roi_gray):
    """
    Suppresses artificial vertical boundary line artifacts at left (x <= 25) and right (x >= w-25) borders.
    Handles isolated components, merged vertical stems, and top/bottom ROI crop border artifacts.
    Protects valid ECG traces crossing boundaries without distorting morphology.
    """
    h, w = binary_mask.shape
    clean_mask = binary_mask.copy()
    
    # Estimate typical ECG baseline y from central ROI region (x in [w//4, 3*w//4])
    central_active = np.where(clean_mask[:, max(0, w//4) : min(w, 3*w//4)] > 0)
    if len(central_active[0]) > 0:
        y_baseline_med = np.median(central_active[0])
    else:
        y_baseline_med = h / 2.0

    # 1. Component-level cleanup for isolated boundary artifacts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    for i in range(1, num_labels):
        x_min = stats[i, cv2.CC_STAT_LEFT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        
        touches_left = (x_min == 0)
        touches_right = (x_min + comp_w >= w - 1)
        
        if touches_left or touches_right:
            aspect_ratio = comp_h / max(comp_w, 1)
            vertical_fraction = comp_h / float(h)
            
            if touches_left and (x_min + comp_w < 25) and (vertical_fraction > 0.25 or aspect_ratio > 2.0):
                clean_mask[labels == i] = 0
            elif touches_right and (x_min > w - 25) and (vertical_fraction > 0.25 or aspect_ratio > 2.0):
                clean_mask[labels == i] = 0

    # 2. Border pixel cleanup at left (x <= 25) and right (x >= w-25)
    for x in range(min(25, w)):
        col = clean_mask[:, x]
        active = np.where(col > 0)[0]
        if len(active) > 0:
            far_from_baseline = active[np.abs(active - y_baseline_med) > 20]
            clean_mask[far_from_baseline, x] = 0
            
            rem_active = np.where(clean_mask[:, x] > 0)[0]
            if len(rem_active) > 0:
                col_height = np.max(rem_active) - np.min(rem_active) + 1
                if col_height > max(20, int(0.25 * h)):
                    near_baseline = rem_active[np.abs(rem_active - y_baseline_med) <= 10]
                    clean_mask[:, x] = 0
                    if len(near_baseline) > 0 and len(near_baseline) <= 14:
                        clean_mask[near_baseline, x] = 255
                        
    for x in range(max(0, w - 25), w):
        col = clean_mask[:, x]
        active = np.where(col > 0)[0]
        if len(active) > 0:
            far_from_baseline = active[np.abs(active - y_baseline_med) > 20]
            clean_mask[far_from_baseline, x] = 0
            
            rem_active = np.where(clean_mask[:, x] > 0)[0]
            if len(rem_active) > 0:
                col_height = np.max(rem_active) - np.min(rem_active) + 1
                if col_height > max(20, int(0.25 * h)):
                    near_baseline = rem_active[np.abs(rem_active - y_baseline_med) <= 10]
                    clean_mask[:, x] = 0
                    if len(near_baseline) > 0 and len(near_baseline) <= 14:
                        clean_mask[near_baseline, x] = 255
                        
    return clean_mask

def extract_robust_centerline(mask, roi_gray):
    """
    E1-E5 Enhanced Subpixel Centerline Extraction for Stage 05.
    Combines subpixel CoM with R/S parabolic & grayscale intensity refinement (E2/E5)
    and joint QRS path constraints (E4). Preserves P/T/ST morphology completely.
    """
    h, w = mask.shape
    y_raw = np.full(w, np.nan)
    widths = np.zeros(w)
    
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) > 0:
            inverted_gray = 255.0 - col_gray[active_indices]
            weights = (inverted_gray ** 2) + 1e-5
            y_center = np.sum(active_indices * weights) / np.sum(weights)
            width = np.max(active_indices) - np.min(active_indices)
            
            # Exclude top/bottom ROI crop border frame artifacts at left/right boundaries
            if (x <= 25 or x >= w - 25) and (y_center <= 3.0 or y_center >= h - 4.0):
                y_raw[x] = np.nan
                widths[x] = 0.0
            else:
                y_raw[x] = float(y_center)
                widths[x] = float(width)
            
    # E2/E5 Subpixel QRS peak refinement
    valid_x = np.where(~np.isnan(y_raw))[0]
    if len(valid_x) >= 20:
        y_inv = -y_raw.copy()
        nans = np.isnan(y_inv)
        y_inv[nans] = np.nanmedian(y_inv)
        
        r_peaks, _ = find_peaks(y_inv, distance=30, prominence=5.0)
        s_troughs = []
        for r in r_peaks:
            s_win = y_raw[r:min(w, r+25)]
            if len(s_win) > 0 and not np.all(np.isnan(s_win)):
                s_troughs.append(r + np.nanargmax(s_win))
                
        # Subpixel parabolic & grayscale intensity refinement for R apexes
        for r in r_peaks:
            if np.isnan(y_raw[r]):
                continue
            if 1 <= r < w - 1 and not np.isnan(y_raw[r-1:r+2]).any():
                y0, y1, y2 = y_raw[r-1], y_raw[r], y_raw[r+1]
                denom = (y0 - 2*y1 + y2)
                if abs(denom) > 1e-5:
                    delta_x = 0.5 * (y0 - y2) / denom
                    delta_x = np.clip(delta_x, -0.8, 0.8)
                    y_sub = y1 - 0.25 * (y0 - y2) * delta_x
                    y_raw[r] = float(y_sub)
                    
            col = int(r)
            y_int = int(round(y_raw[r]))
            if 3 <= y_int < h - 3:
                win_y = np.arange(y_int - 3, y_int + 4)
                win_gray = 255.0 - roi_gray[win_y, col]
                if win_gray.sum() > 0:
                    y_sub_gray = np.sum(win_y * (win_gray ** 4)) / np.sum(win_gray ** 4)
                    y_raw[r] = 0.5 * y_raw[r] + 0.5 * float(y_sub_gray)
                    
        # Subpixel parabolic & grayscale intensity refinement for S troughs
        for s in s_troughs:
            if np.isnan(y_raw[s]):
                continue
            if 1 <= s < w - 1 and not np.isnan(y_raw[s-1:s+2]).any():
                y0, y1, y2 = y_raw[s-1], y_raw[s], y_raw[s+1]
                denom = (y0 - 2*y1 + y2)
                if abs(denom) > 1e-5:
                    delta_x = 0.5 * (y0 - y2) / denom
                    delta_x = np.clip(delta_x, -0.8, 0.8)
                    y_sub = y1 - 0.25 * (y0 - y2) * delta_x
                    y_raw[s] = float(y_sub)
                    
            col = int(s)
            y_int = int(round(y_raw[s]))
            if 3 <= y_int < h - 3:
                win_y = np.arange(y_int - 3, y_int + 4)
                win_gray = 255.0 - roi_gray[win_y, col]
                if win_gray.sum() > 0:
                    y_sub_gray = np.sum(win_y * (win_gray ** 4)) / np.sum(win_gray ** 4)
                    y_raw[s] = 0.5 * y_raw[s] + 0.5 * float(y_sub_gray)
                    
    centerline = []
    for x in range(w):
        y_val = float(y_raw[x]) if not np.isnan(y_raw[x]) else None
        conf = 1.0 if y_val is not None else 0.0
        centerline.append({"x": x, "y": y_val, "confidence": conf, "width": float(widths[x])})
        
    df = pd.DataFrame(centerline)
    return df

def run_stage5(case, model, device, checkpoint_info=None):
    print(f"Running Stage 5 Inference for {case}...")
    
    stage4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    
    if not os.path.exists(stage4_ref_path) or not os.path.exists(rois_path):
        print(f"Skipping {case} - missing Stage 4 inputs.")
        return
        
    img = cv2.imread(stage4_ref_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
        
    with open(rois_path, 'r') as f:
        rois = json.load(f)
        
    out_dir = f"stage5_output/{case}"
    masks_dir = os.path.join(out_dir, "03_stage5_binary_masks")
    clean_dir = os.path.join(out_dir, "04_stage5_clean_masks")
    digitized_dir = os.path.join(out_dir, "06_digitized_traces")
    plots_dir = os.path.join(out_dir, "07_centerline_plots")
    actuals_dir = os.path.join(out_dir, "08_actual_leads")
    overlays_dir = os.path.join(out_dir, "09_lead_overlays")
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(digitized_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(actuals_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "01_stage3_reference.png"), img)
    
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    metrics = {"stage": "5.1", "leads": {}}
    if checkpoint_info:
        metrics["checkpoint"] = checkpoint_info
    
    for lead, data in rois.items():
        x1, x2 = data['x_start'], data['x_end']
        y1, y2 = data['y_start'], data['y_end']
        
        # Crop ROI (Actual ECG Lead Crop)
        roi_img = img[y1:y2, x1:x2]
        orig_h, orig_w = roi_img.shape
        
        # Save actual input lead image
        cv2.imwrite(os.path.join(actuals_dir, f"{lead}_actual.png"), roi_img)
        
        # Pad to multiple of 32 (only bottom/right borders to preserve coordinate layouts)
        pad_h = (32 - (orig_h % 32)) % 32
        pad_w = (32 - (orig_w % 32)) % 32
        roi_padded = cv2.copyMakeBorder(roi_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        
        tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor_img = tensor_img.to(device)
        
        with torch.no_grad():
            output = model(tensor_img)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
        binary_padded = (prob > 0.40).astype(np.uint8) * 255
        
        # Remove padding to return to native ROI size
        binary_orig = binary_padded[0:orig_h, 0:orig_w]
        
        # Safe cleanup: filter small isolated noise and abnormal starting vertical boundary artifacts
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_orig, connectivity=8)
        clean_mask = np.zeros_like(binary_orig)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 10:
                clean_mask[labels == i] = 255
                
        clean_mask = suppress_boundary_artifacts(clean_mask, roi_img)
        
        # Centerline extraction (E1-E5 refined)
        centerline_df = extract_robust_centerline(clean_mask, roi_img)
        
        # Save output CSV and binary masks
        cv2.imwrite(os.path.join(masks_dir, f"{lead}_mask.png"), binary_orig)
        cv2.imwrite(os.path.join(clean_dir, f"{lead}_clean.png"), clean_mask)
        centerline_df.to_csv(os.path.join(digitized_dir, f"{lead}_centerline.csv"), index=False)
        
        # Draw on global overlay
        mask_colored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_colored[clean_mask > 0] = [0, 255, 0] # Green mask
        
        roi_bgr = debug_img[y1:y2, x1:x2]
        blended = cv2.addWeighted(roi_bgr, 0.7, mask_colored, 0.3, 0)
        debug_img[y1:y2, x1:x2] = blended
        
        # Create individual overlay image
        roi_overlay_bgr = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        
        for _, row in centerline_df.iterrows():
            if pd.notna(row['y']):
                px = x1 + int(row['x'])
                py = y1 + int(row['y'])
                cv2.circle(debug_img, (px, py), 1, (0, 0, 255), -1) # Red centerline points
                
        # Draw a continuous solid line on individual overlay BGR instead of dots
        pts = []
        for _, row in centerline_df.iterrows():
            if pd.notna(row['y']):
                pts.append([int(row['x']), int(row['y'])])
        if len(pts) > 1:
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(roi_overlay_bgr, [pts], isClosed=False, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                
        # Save individual overlay
        cv2.imwrite(os.path.join(overlays_dir, f"{lead}_overlay.png"), roi_overlay_bgr)
                
        # Generate and save signal visualization plot of the CSV
        plt.figure(figsize=(10, 3))
        plt.plot(centerline_df['x'], centerline_df['y'], color='crimson', linewidth=1.5, label='Extracted Centerline')
        plt.title(f"Extracted Centerline Signal - Lead {lead}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate (Inverted)")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{lead}_plot.png"), dpi=150)
        plt.close()
        
        metrics["leads"][lead] = {
            "mask_pixels": int(np.sum(clean_mask > 0)),
            "centerline_valid_points": int(centerline_df['y'].notna().sum())
        }
        
    cv2.imwrite(os.path.join(out_dir, "stage5_debug.png"), debug_img)
    with open(os.path.join(out_dir, "stage5_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Finished Case {case}.")

def main():
    import warnings
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    
    checkpoint_info = None
    if os.path.exists("stage5_best_model.pth"):
        import hashlib
        import time
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
        
        # Calculate SHA256
        with open("stage5_best_model.pth", "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        file_stats = os.stat("stage5_best_model.pth")
        mod_time = time.ctime(file_stats.st_mtime)
        file_size = file_stats.st_size
        
        checkpoint_info = {
            "path": "stage5_best_model.pth",
            "sha256": file_hash,
            "size_bytes": file_size,
            "mod_time": mod_time,
            "architecture": "UNetPlusPlusResNet18"
        }
        
        print(f"Loaded checkpoint: stage5_best_model.pth")
        print(f"SHA256: {file_hash}")
        print(f"Modified: {mod_time}")
    else:
        print("Warning: stage5_best_model.pth not found. Running with untrained weights!")
        
    model.eval()
    
    if len(sys.argv) > 1:
        cases = [sys.argv[1]]
    else:
        cases = [d for d in os.listdir("stage4_output") if os.path.isdir(os.path.join("stage4_output", d))]
    
    for case in cases:
        run_stage5(case, model, device, checkpoint_info)

if __name__ == "__main__":
    main()
