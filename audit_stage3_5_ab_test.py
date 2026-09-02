import os
import cv2
import numpy as np
import json
import glob
from collections import defaultdict
import mlflow

from ecg_digitizer.modules.stage03_grid_removal import run_stage3
import ecg_digitizer.modules.stage03_5_glyph_disentanglement as stage3_5

def get_stage5_clean_mask(case_name, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_dir = os.path.join("stage5_output", case_name, "04_stage5_clean_masks")
    if not os.path.exists(mask_dir):
        return mask
    for file in os.listdir(mask_dir):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                mask = cv2.bitwise_or(mask, img)
    return mask

def get_rois(case_name):
    roi_file = os.path.join("stage4_output", case_name, "lead_rois.json")
    if not os.path.exists(roi_file):
        return {}
    with open(roi_file, 'r') as f:
        return json.load(f)

def run_audit(disable_e84=False):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ECG_Digitization")
    
    test_cases = ["Copy of 19_male", "Copy of 55_male", "Copy of 60_male"]
    
    with mlflow.start_run(run_name=f"Forensic_AB_Test_{'D6_E84_OFF' if disable_e84 else 'D2_Full'}"):
        
        mlflow.log_param("disable_e84", disable_e84)
        
        total_ecg_pixels_lost = 0
        total_waveform_changes = 0
        
        for case_name in test_cases:
            image_path = os.path.join("stage2_2_output", case_name, "01_panel.png")
            if not os.path.exists(image_path):
                continue
                
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            metrics_path = os.path.join("ab_test_out", case_name, "metrics.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            
            # Run the integrated stage 3 pipeline
            outputs, diags, metrics = run_stage3(image, metrics_path)
            
            s3_ecg_img = diags.get("09_stage3_final_clean_ecg")
            s35_ecg_img = outputs.get("09_final_clean_ecg")
            
            if s3_ecg_img is None or s35_ecg_img is None:
                continue
                
            s3_mask = (s3_ecg_img[:, :, 0] == 0).astype(np.uint8) * 255
            s35_mask = (s35_ecg_img[:, :, 0] == 0).astype(np.uint8) * 255
            
            # Difference Mask: pixels that WERE in S3 but ARE NOT in S3.5 (i.e., Deleted by S3.5)
            diff_mask = cv2.bitwise_and(s3_mask, cv2.bitwise_not(s35_mask))
            
            # Intersect with Stage 5 ground-truth equivalent
            stage5_mask = get_stage5_clean_mask(case_name, h, w)
            if np.sum(stage5_mask) > 0:
                damaged_ecg_pixels = cv2.bitwise_and(diff_mask, stage5_mask)
            else:
                # If stage5 mask doesn't exist, use S3 mask as proxy for "any foreground deleted"
                damaged_ecg_pixels = diff_mask
            
            damaged_count = int(np.sum(damaged_ecg_pixels > 0))
            total_waveform_changes += damaged_count
            
            print(f"[{case_name}] Waveform Pixels Damaged by Stage 3.5: {damaged_count}")
            
            # Spatial localization
            rois = get_rois(case_name)
            for lead, coords in rois.items():
                x1, y1, x2, y2 = coords
                lead_damage = np.sum(damaged_ecg_pixels[y1:y2, x1:x2] > 0)
                if lead_damage > 0:
                    print(f"   -> {lead} ROI: {lead_damage} pixels damaged")

            cv2.imwrite(os.path.join("ab_test_out", case_name, f"diff_mask_{'D6' if disable_e84 else 'D2'}.png"), diff_mask)
            cv2.imwrite(os.path.join("ab_test_out", case_name, f"damaged_ecg_{'D6' if disable_e84 else 'D2'}.png"), damaged_ecg_pixels)
            
        mlflow.log_metric("total_waveform_pixels_damaged", total_waveform_changes)
        print(f"Total waveform pixels damaged across cases: {total_waveform_changes}")
        
        # Determine culprit
        if total_waveform_changes > 0:
            print("FAIL: Stage 3.5 deleted true ECG pixels. Invariant violated!")
        else:
            print("PASS: Stage 3.5 preserved all ECG morphology pixels.")

if __name__ == "__main__":
    os.makedirs("ab_test_out", exist_ok=True)
    print("Running D2 Experiment (Full Stage 3.5)...")
    run_audit(disable_e84=False)
