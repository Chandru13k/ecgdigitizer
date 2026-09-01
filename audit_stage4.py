import cv2
import numpy as np
import json
import os
import sys

def audit_case(case):
    print(f"\n--- Auditing Stage 4 (Valley-Finding Bounds): {case} ---")
    stg3_path = f"stage3_output/{case}/09_final_clean_ecg.png"
    stg4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    
    if not os.path.exists(stg3_path) or not os.path.exists(stg4_ref_path) or not os.path.exists(rois_path):
        print(f"Skipping {case} - missing files.")
        return False
        
    stg3_img = cv2.imread(stg3_path, 0)
    stg4_img = cv2.imread(stg4_ref_path, 0)
    
    _, stg3_bin = cv2.threshold(stg3_img, 128, 255, cv2.THRESH_BINARY_INV)
    _, stg4_bin = cv2.threshold(stg4_img, 128, 255, cv2.THRESH_BINARY_INV)
    
    with open(rois_path, 'r') as f:
        rois = json.load(f)
        
    all_pass = True
        
    print(f"1. Pixel Modification Check:")
    diff = cv2.absdiff(stg3_img, stg4_img)
    changed_pixels = np.sum(diff > 0)
    if changed_pixels > 0:
        print(f"   FAIL: Stage 4 modified {changed_pixels} pixels.")
        all_pass = False
    else:
        print("   PASS: 0 pixels modified. Stage 4 reference is pixel-identical to Stage 3.")
        
    print(f"2. Number of ROIs Check:")
    if len(rois) != 13:
        print(f"   FAIL: Expected 13 ROIs, found {len(rois)}.")
        all_pass = False
    else:
        print("   PASS: 13 ROIs generated.")
        
    print(f"3. ROI Topology (Overlap = 0) Check:")
    assigned_mask = np.zeros_like(stg3_bin, dtype=np.uint8)
    overlap_detected = False
    
    for lead, data in rois.items():
        x1, x2 = data['x_start'], data['x_end']
        y1, y2 = data['y_start'], data['y_end']
        
        # Check if any pixels in this ROI have already been assigned (Overlap)
        roi_mask = np.zeros_like(stg3_bin, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        
        overlap_pixels = cv2.bitwise_and(assigned_mask, roi_mask)
        if np.sum(overlap_pixels > 0) > 0:
            print(f"   FAIL: ROI for {lead} overlaps geometrically with a previously assigned ROI.")
            overlap_detected = True
            all_pass = False
            
        assigned_mask = cv2.bitwise_or(assigned_mask, roi_mask)
        
    if not overlap_detected:
        print("   PASS: ROI geometric overlap = 0. All boundaries are strictly distinct.")
        
    print(f"4. Orphaned Pixel Check:")
    orphaned_pixels = cv2.bitwise_and(stg3_bin, cv2.bitwise_not(assigned_mask))
    orphan_count = np.sum(orphaned_pixels > 0)
    
    # We must categorize orphans. Are they true ECG that fell off the image, or isolated noise?
    if orphan_count > 0:
        # A true deep QRS clipping would create large contiguous orphan blocks
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(orphaned_pixels, connectivity=8)
        
        large_orphans = 0
        for i in range(1, num_labels): # 0 is background
            if stats[i, cv2.CC_STAT_AREA] > 50: # If there's a big chunk of waveform missing
                large_orphans += stats[i, cv2.CC_STAT_AREA]
                
        if large_orphans > 0:
            print(f"   FAIL: {orphan_count} foreground pixels unassigned. {large_orphans} pixels appear to be substantial clipped physiological waveform.")
            all_pass = False
        else:
            print(f"   PASS WITH WARNING: {orphan_count} foreground pixels unassigned, but they are all small isolated artifacts (<50 px). Likely noise.")
    else:
        print("   PASS: 0 foreground pixels unassigned. (Orphans = 0)")

    return all_pass

def main():
    cases = ['Copy of 19_male', 'Copy of 20_male']
    all_pass = True
    
    for case in cases:
        if not audit_case(case):
            all_pass = False
            
    if all_pass:
        print("\n============================================================")
        print("STAGE 4 VALIDATED — READY FOR STAGE 5 SEMANTIC WAVEFORM SEGMENTATION.")
        print("============================================================")
    else:
        print("\nSTAGE 4 VALIDATION FAILED.")

if __name__ == "__main__":
    main()
