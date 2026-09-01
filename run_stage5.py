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

def extract_robust_centerline(mask, roi_gray):
    """
    Extracts 1D centerline from binary mask using subpixel grayscale weighting (I^2).
    """
    h, w = mask.shape
    centerline = []
    
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) > 0:
            inverted_gray = 255.0 - col_gray[active_indices]
            weights = (inverted_gray ** 2) + 1e-5
            y_center = np.sum(active_indices * weights) / np.sum(weights)
            width = np.max(active_indices) - np.min(active_indices)
            centerline.append({"x": x, "y": float(y_center), "confidence": 1.0, "width": float(width)})
        else:
            centerline.append({"x": x, "y": None, "confidence": 0.0, "width": 0.0})
            
    df = pd.DataFrame(centerline)
    return df

def run_stage5(case, model, device):
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
        
        # Safe cleanup: only filter extremely small isolated components to prevent losing thin QRS/peaks
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_orig, connectivity=8)
        clean_mask = np.zeros_like(binary_orig)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 10:
                clean_mask[labels == i] = 255
                
        # Centerline extraction
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
    
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    else:
        print("Warning: stage5_best_model.pth not found. Running with untrained weights!")
        
    model.eval()
    
    if len(sys.argv) > 1:
        cases = [sys.argv[1]]
    else:
        cases = [d for d in os.listdir("stage4_output") if os.path.isdir(os.path.join("stage4_output", d))]
    
    for case in cases:
        run_stage5(case, model, device)

if __name__ == "__main__":
    main()
