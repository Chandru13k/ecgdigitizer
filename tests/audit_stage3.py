import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m2_preprocessing import preprocess_image
from modules.m3_geometry_correction import correct_skew_and_geometry
from modules.m6_grid_removal import remove_grid

def audit_stage3():
    target_leads = [
        ("20_male", ["V2", "V3", "aVR", "aVF"]),
        ("19_male", ["V2", "V3"])
    ]
    
    # We will test on the full image because Stage 3 runs on the full aligned image
    # We will just visually and quantitatively audit the global removal.
    
    out_dir = "tests/stage3_audits"
    os.makedirs(out_dir, exist_ok=True)
    
    total_vert_removed = 0
    total_grid_reduction = 0
    
    import glob
    for img_path in sorted(glob.glob("stage2_2_output/Copy of */03_ecg_panel_crop.png")):
        case_name = os.path.basename(os.path.dirname(img_path))
        aligned_color = cv2.imread(img_path)
        
        res = remove_grid(aligned_color, {})
        
        ink = res["ink_candidate"]
        grid = res["grid_candidate"]
        glob_sep = res["global_separators"]
        protected = res["waveform_protected"]
        
        # Save debug images
        cv2.imwrite(f"{out_dir}/{case_name}_01_input.png", aligned_color)
        cv2.imwrite(f"{out_dir}/{case_name}_02_ink.png", ink)
        cv2.imwrite(f"{out_dir}/{case_name}_03_grid_candidate.png", grid)
        cv2.imwrite(f"{out_dir}/{case_name}_04_global_separators.png", glob_sep)
        cv2.imwrite(f"{out_dir}/{case_name}_05_waveform_protected.png", protected)
        
        # Overlay
        overlay = aligned_color.copy()
        # Green = protected ECG
        # Red = removed global separators
        overlay[glob_sep > 0] = (0, 0, 255)
        # overlay[protected > 0] = (0, 255, 0)
        cv2.imwrite(f"{out_dir}/{case_name}_06_overlay_removed.png", overlay)
        
        glob_pixels = np.sum(glob_sep > 0)
        ink_pixels = np.sum(ink > 0)
        protected_pixels = np.sum(protected > 0)
        
        print(f"Case: {case_name}")
        print(f"  Total Ink Pixels: {ink_pixels}")
        print(f"  Global Separator Pixels Removed: {glob_pixels}")
        print(f"  Protected Pixels: {protected_pixels}")
        print(f"  % Separator Removed: {(glob_pixels/ink_pixels)*100:.1f}%")
        
        total_vert_removed += glob_pixels
        
if __name__ == "__main__":
    audit_stage3()
