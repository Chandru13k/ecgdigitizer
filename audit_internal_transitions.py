import os
import cv2
import numpy as np
import json
import mlflow

from ecg_digitizer.modules.stage03_grid_removal import run_stage3

def get_stage5_clean_mask(case_name, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_dir = os.path.join("stage5_output", case_name, "04_stage5_clean_masks")
    if not os.path.exists(mask_dir):
        return mask
    for file in os.listdir(mask_dir):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != mask.shape:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_or(mask, img)
    return mask

def get_rois(case_name):
    roi_file = os.path.join("stage4_output", case_name, "lead_rois.json")
    if not os.path.exists(roi_file):
        return {}
    with open(roi_file, 'r') as f:
        return json.load(f)

def intersect_and_report(case_name, name, before_mask, after_mask, pristine_mask, rois, out_dir, h, w):
    # Mask format: 255 = foreground ECG pixel
    # Difference: Pixels in before_mask that are missing in after_mask
    diff_mask = cv2.bitwise_and(before_mask, cv2.bitwise_not(after_mask))
    
    if np.sum(pristine_mask) > 0:
        damaged_ecg_pixels = cv2.bitwise_and(diff_mask, pristine_mask)
    else:
        damaged_ecg_pixels = diff_mask
        
    damaged_count = int(np.sum(damaged_ecg_pixels > 0))
    print(f"  [{name}] Waveform Pixels Damaged: {damaged_count}")
    
    if damaged_count > 0:
        cv2.imwrite(os.path.join(out_dir, f"{case_name}_{name}_diff.png"), damaged_ecg_pixels)
        for lead, coords_dict in rois.items():
            x1 = coords_dict.get("x_start", 0)
            x2 = coords_dict.get("x_end", w)
            y1 = coords_dict.get("y_start", 0)
            y2 = coords_dict.get("y_end", h)
            lead_damage = np.sum(damaged_ecg_pixels[y1:y2, x1:x2] > 0)
            if lead_damage > 0:
                print(f"    -> {lead} ROI: {lead_damage} pixels damaged")
    
    return damaged_count

def run_internal_audit():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ECG_Digitization")
    
    test_cases = ["Copy of 19_male", "Copy of 55_male", "Copy of 60_male"]
    out_dir = "ab_test_out_internal"
    os.makedirs(out_dir, exist_ok=True)
    
    with mlflow.start_run(run_name="Internal_Checkpoints_Audit"):
        
        total_c0_c1 = 0
        total_c1_c2 = 0
        total_c2_c3 = 0
        
        for case_name in test_cases:
            print(f"\nProcessing {case_name}...")
            image_path = os.path.join("stage2_2_output", case_name, "01_original.png")
            if not os.path.exists(image_path):
                print(f"  Missing image: {image_path}")
                continue
                
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            stage2_metrics_path = os.path.join("stage2_2_output", case_name, "stage2_2_metrics.json")
            
            case_out_dir = os.path.join(out_dir, case_name)
            os.makedirs(case_out_dir, exist_ok=True)
            
            # Run the integrated stage 3 pipeline
            results = run_stage3(image, stage2_metrics_path)
            
            # Retrieve Checkpoints
            c0 = results.get("C0_candidate")
            c1 = results.get("C1_post_isolated_label")
            c2 = results.get("C2_post_overlap_label")
            
            if c0 is None or c1 is None or c2 is None:
                print("  Checkpoints not found in diagnostic_images!")
                continue
                
            # C3 logic: physical save and reload of the final image
            final_img = results.get("09_final_clean_ecg")
            c3_path = os.path.join(out_dir, case_name, "C3_reloaded.png")
            cv2.imwrite(c3_path, final_img)
            c3_reloaded = cv2.imread(c3_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert all to binary mask (255 = foreground)
            c0_mask = (c0 > 0).astype(np.uint8) * 255
            c1_mask = (c1 > 0).astype(np.uint8) * 255
            c2_mask = (c2 > 0).astype(np.uint8) * 255
            c3_mask = (c3_reloaded == 0).astype(np.uint8) * 255  # black pixels in image are foreground
            
            h, w = c0.shape[:2]
            
            stage5_mask = get_stage5_clean_mask(case_name, h, w)
            rois = get_rois(case_name)
            
            # Compare transitions
            d1 = intersect_and_report(case_name, "C0_to_C1_baseline", c0_mask, c1_mask, stage5_mask, rois, out_dir, h, w)
            d2 = intersect_and_report(case_name, "C1_to_C2_stage3_5", c1_mask, c2_mask, stage5_mask, rois, out_dir, h, w)
            d3 = intersect_and_report(case_name, "C2_to_C3_physical_io", c2_mask, c3_mask, stage5_mask, rois, out_dir, h, w)
            
            total_c0_c1 += d1
            total_c1_c2 += d2
            total_c2_c3 += d3
            
        mlflow.log_metrics({
            "total_damage_C0_C1": total_c0_c1,
            "total_damage_C1_C2": total_c1_c2,
            "total_damage_C2_C3": total_c2_c3
        })
        
        print(f"\nFinal Summary across all cases:")
        print(f"  C0->C1 (Stage 3 Baseline Subtraction) damage: {total_c0_c1} pixels")
        print(f"  C1->C2 (Stage 3.5 Disentanglement) damage: {total_c1_c2} pixels")
        print(f"  C2->C3 (Physical Save/Reload IO) damage: {total_c2_c3} pixels")
        
        if total_c1_c2 > 0:
            print("CULPRIT: Stage 3.5 Disentanglement!")
        elif total_c2_c3 > 0:
            print("CULPRIT: Image saving/loading quantization/compression!")
        elif total_c0_c1 > 0:
            print("CULPRIT: Original Stage 3 (C0->C1) subtraction/geometry logic!")
        else:
            print("No damage detected in Stage 3 across any checkpoint.")

if __name__ == "__main__":
    print("Running Internal Checkpoint Audit (C0->C1->C2->C3)...")
    run_internal_audit()
