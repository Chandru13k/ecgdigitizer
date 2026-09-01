import cv2
import json
import os
from ecg_digitizer.modules.stage03_grid_removal import run_stage3

def main():
    print("=========================================================")
    print("STAGE 3 \u2014 GRID REMOVAL + GLOBAL PERIODIC SEPARATOR REMOVAL")
    print("=========================================================")
    
    cases = [d for d in os.listdir("stage2_2_output") if os.path.isdir(os.path.join("stage2_2_output", d))]
    
    for case in sorted(cases):
        print(f"\n{'='*60}")
        print(f"Processing {case}...")
        print(f"{'='*60}")
        
        original_img_path = f"stage2_2_output/{case}/01_original.png"
        metrics_path = f"stage2_2_output/{case}/stage2_2_metrics.json"
        
        if not os.path.exists(original_img_path) or not os.path.exists(metrics_path):
            print(f"Skipping {case} - missing inputs.")
            continue
            
        image = cv2.imread(original_img_path)
        if image is None:
            print(f"Error loading {original_img_path}")
            continue
            
        out_dir = f"stage3_output/{case}"
        os.makedirs(out_dir, exist_ok=True)
        
        results = run_stage3(image, metrics_path)
        
        # Save core outputs
        cv2.imwrite(f"{out_dir}/01_active_panel.png", results["01_active_panel"])
        cv2.imwrite(f"{out_dir}/02_grid_mask.png", results["02_grid_mask"])
        cv2.imwrite(f"{out_dir}/03_grid_removed.png", results["03_grid_removed"])
        cv2.imwrite(f"{out_dir}/04_waveform_mask.png", results["04_waveform_mask"])
        cv2.imwrite(f"{out_dir}/05_clean_ecg.png", results["05_clean_ecg"])
        cv2.imwrite(f"{out_dir}/06_grid_removal_overlay.png", results["06_grid_removal_overlay"])
        
        # Save Stage 3.1 diagnostic outputs
        for key in ["07_01_candidate_mask", "07_02_separator_mask",
                     "07_03_calibration_mask", "07_04_text_mask",
                     "07_05_artifact_mask", "07_06_final_ecg_mask",
                     "07_07_subtraction_overlay",
                     "07_08_separator_intersection_with_ecg",
                     "07_09_calibration_intersection_with_ecg",
                     "07_10_text_intersection_with_ecg",
                     "07_11_lost_ecg_pixels",
                     "07_12_waveform_preservation_overlay",
                     "07_13_waveform_at_separator",
                     "07_14_interpolated_pixels"]:
            img = results.get(key)
            if img is not None:
                cv2.imwrite(f"{out_dir}/{key}.png", img)
        
        # Save final outputs
        cv2.imwrite(f"{out_dir}/08_final_ecg_mask.png", results["08_final_ecg_mask"])
        cv2.imwrite(f"{out_dir}/09_final_clean_ecg.png", results["09_final_clean_ecg"])
        
        # Save new diagnostic outputs (Steps 10-12)
        cv2.imwrite(f"{out_dir}/10_separator_positions.png", results["10_separator_positions"])
        cv2.imwrite(f"{out_dir}/11_separator_mask.png", results["11_separator_mask"])
        
        # Save E24 residual map and E36 visual report figures
        if "07_15_residual_separator_map" in results:
            cv2.imwrite(f"{out_dir}/12_residual_separator_map.png", results["07_15_residual_separator_map"])
        else:
            cv2.imwrite(f"{out_dir}/12_difference_image.png", results["12_difference_image"])
            
        if "07_16_candidates_overlay" in results:
            cv2.imwrite(f"{out_dir}/02_candidates_overlay.png", results["07_16_candidates_overlay"])
            
        if "07_17_before_after_overlay" in results:
            cv2.imwrite(f"{out_dir}/13_before_after_overlay.png", results["07_17_before_after_overlay"])

        # E26 to E31 visual diagnostics
        if "07_02_separator_mask" in results:
            cv2.imwrite(f"{out_dir}/14_separator_mask.png", results["07_02_separator_mask"])
        if "07_18_removed_pixel_map" in results:
            cv2.imwrite(f"{out_dir}/15_removed_pixel_map.png", results["07_18_removed_pixel_map"])
        if "07_19_separator_crops" in results:
            cv2.imwrite(f"{out_dir}/16_separator_crops.png", results["07_19_separator_crops"])
            cv2.imwrite(f"{out_dir}/17_before_after_separator.png", results["07_19_separator_crops"])
        if "07_20_candidate_decision_map" in results:
            cv2.imwrite(f"{out_dir}/18_candidate_decision_map.png", results["07_20_candidate_decision_map"])
        if "07_21_residual_localization" in results:
            cv2.imwrite(f"{out_dir}/19_residual_localization.png", results["07_21_residual_localization"])
        if "07_22_removed_vs_preserved" in results:
            cv2.imwrite(f"{out_dir}/20_removed_vs_preserved.png", results["07_22_removed_vs_preserved"])
        
        metrics = results["metrics"]
        with open(f"{out_dir}/stage3_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Console summary
        print(f"\nInput panel: {metrics['active_panel_width']} x {metrics['active_panel_height']}")
        
        print(f"\nSeparator Detection:")
        print(f"  Count          : {metrics['number_of_vertical_lines_detected']}")
        print(f"  Spacing        : {metrics.get('estimated_separator_spacing', 0):.1f} px")
        print(f"  Periodicity    : {metrics.get('periodicity_confidence', 0):.2f}")
        print(f"  X-coordinates  : {metrics.get('separator_x_coordinates', [])}")
        print(f"  Confidences    : {metrics.get('separator_confidences', [])}")
        
        if metrics['number_of_vertical_lines_detected'] > 0:
            print(f"\n  | # | X | Confidence |")
            print(f"  |---|---|---|")
            for idx, line in enumerate(metrics['vertical_lines_details']):
                print(f"  | {idx+1} | {line.get('x_coordinate', 0)} | {line.get('confidence', 0):.4f} |")
                
        print(f"\n  Calibration Conf: {metrics['calibration_detection_confidence']:.2f}")
        
        print(f"\nLabel Removal:")
        label_det = metrics.get('label_retention_details', {})
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            if lead in label_det:
                print(f"  {lead:<4} : {label_det[lead]}")
                
        print(f"\nCharacter Removal Audit (E5):")
        print(f"  Expected Labels       : {metrics.get('char_removal_expected', 12)}")
        print(f"  Detected Before       : {metrics.get('char_removal_detected_before', 0)}")
        print(f"  Detected After        : {metrics.get('char_removal_detected_after', 0)}")
        print(f"  Residual Label Pixels : {metrics.get('char_removal_residual_pixels', 0)}")
        print(f"  Status                : {metrics.get('char_removal_status', 'FAIL')}")
        
        print(f"\nPixel Retention:")
        print(f"  Candidate ECG pixels       : {metrics['candidate_ecg_pixels']}")
        print(f"  ECG Pixels Preserved       : {metrics.get('ecg_pixels_preserved', 0)}")
        print(f"  ECG Pixels Lost            : {metrics.get('ecg_pixels_lost', 0)}")
        print(f"  Separator pixels removed   : {metrics['separator_pixels_removed']}")
        print(f"  Residual separator pixels  : {metrics.get('residual_separator_pixels', 0)}")
        print(f"  Width variance across leads: {metrics.get('width_variance_across_leads', 0.0)}")
        print(f"  Mistaken waveform count    : {metrics.get('mistaken_waveform_as_separator', 0)}")
        print(f"  Calibration pixels removed : {metrics['calibration_pixels_removed']}")
        print(f"  Text pixels removed        : {metrics.get('text_pixels_removed', 0)}")
        print(f"  Interpolated pixels added  : {metrics.get('interpolated_pixels_added', 0)}")
        print(f"  Total pixels removed       : {metrics['total_pixels_removed']}")
        print(f"  Final ECG pixels           : {metrics['final_ecg_pixels']}")
        print(f"  Loss ratio                 : {metrics.get('ecg_pixel_loss_ratio', 0)*100:.2f}%")
        
        print(f"\nPer-Lead Waveform Preservation:")
        print(f"  | Lead | Cand Px | Fin Px | Removed | Retention | Cand Amp | Fin Amp |")
        print(f"  |---|---|---|---|---|---|---|")
        per_lead = metrics.get('per_lead_pixel_retention', {})
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            if lead in per_lead:
                d = per_lead[lead]
                print(f"  | {lead:<4} | {d['candidate_pixels']} | {d['final_pixels']} | "
                      f"{d['removed_pixels']} | {d['retention_ratio']*100:.2f}% | "
                      f"{d['geometry_candidate']['amplitude']} | {d['geometry_final']['amplitude']} |")
                
        print(f"\n  Outputs 01-12 generated. stage3_metrics.json updated.")

if __name__ == "__main__":
    main()
