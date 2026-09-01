import os
import json
import cv2
import numpy as np

def generate_pseudo_labels(case):
    print(f"Generating pseudo-labels for {case}...")
    stage3_path = f"stage3_output/{case}/09_final_clean_ecg.png"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    
    if not os.path.exists(stage3_path) or not os.path.exists(rois_path):
        print(f"Skipping {case} - required Stage 3/4 files missing.")
        return
        
    img = cv2.imread(stage3_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
        
    # Threshold: Assuming dark pixels are foreground
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    with open(rois_path, 'r') as f:
        rois = json.load(f)
        
    out_dir = f"pseudo_labels/{case}"
    os.makedirs(out_dir, exist_ok=True)
    
    for lead, data in rois.items():
        x1, x2 = data['x_start'], data['x_end']
        y1, y2 = data['y_start'], data['y_end']
        
        roi_mask = binary[y1:y2, x1:x2]
        
        # Connected components to remove small noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
        
        # Keep components larger than 20 pixels
        clean_mask = np.zeros_like(roi_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 20:
                clean_mask[labels == i] = 255
                
        # To avoid clipping small P waves that might be disconnected due to fading,
        # we can use a slightly smaller threshold, e.g., > 10 pixels.
        # But this is just a pseudo-label. We want it to be as clean as possible.
        # Actually, let's just save the crop itself and the clean_mask.
        
        cv2.imwrite(f"{out_dir}/{lead}_img.png", img[y1:y2, x1:x2])
        cv2.imwrite(f"{out_dir}/{lead}_mask.png", clean_mask)
        
    print(f"Generated pseudo-labels for {len(rois)} leads in {case}.")

def main():
    cases = ['Copy of 19_male', 'Copy of 20_male']
    for case in cases:
        generate_pseudo_labels(case)

if __name__ == "__main__":
    main()
