import sys
import os
import cv2
import json
import shutil
from ecg_digitizer.modules.stage04_lead_segmentation import run_stage4

def main():
    if len(sys.argv) > 1:
        cases = [sys.argv[1]]
    else:
        cases = [d for d in os.listdir("stage3_output") if os.path.isdir(os.path.join("stage3_output", d))]
    
    for case in cases:
        print(f"Running Stage 4 on {case}...")
        
        # Inputs
        stage3_img_path = f"stage3_output/{case}/09_final_clean_ecg.png"
        stage3_metrics_path = f"stage3_output/{case}/stage3_metrics.json"
        stage2_2_metrics_path = f"stage2_2_output/{case}/stage2_2_metrics.json"
        
        if not os.path.exists(stage3_img_path):
            print(f"Skipping {case} - missing Stage 3 output.")
            continue
            
        stage3_img = cv2.imread(stage3_img_path)
        
        with open(stage3_metrics_path, 'r') as f:
            stage3_metrics = json.load(f)
            
        with open(stage2_2_metrics_path, 'r') as f:
            stage2_2_metrics = json.load(f)
            
        # Run Stage 4
        rois, debug_img, metrics = run_stage4(stage3_img, stage3_metrics, stage2_2_metrics)
        
        # Output directory
        out_dir = f"stage4_output/{case}"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save exact copy of Stage 3
        shutil.copy2(stage3_img_path, f"{out_dir}/01_stage3_reference.png")
        
        # Save Debug image
        cv2.imwrite(f"{out_dir}/lead_segmentation_debug.png", debug_img)
        
        # Save ROIs and metrics
        with open(f"{out_dir}/lead_rois.json", 'w') as f:
            json.dump(rois, f, indent=4)
            
        with open(f"{out_dir}/stage4_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Finished {case}. ROIs generated: {metrics['total_leads_detected']}")

if __name__ == "__main__":
    main()
