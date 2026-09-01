import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m6_grid_removal import remove_grid
import glob

def run():
    target_leads = {
        "20_male": ["V2", "V3", "aVR", "aVF"],
        "19_male": ["V2", "V3"]
    }
    
    out_base = "tests/stage3_validation_outputs"
    os.makedirs(out_base, exist_ok=True)
    
    # We will process the full aligned images, but then crop to the requested leads for visualization
    for img_path in sorted(glob.glob("stage2_2_output/Copy of */03_ecg_panel_crop.png")):
        case_name_raw = os.path.basename(os.path.dirname(img_path))
        case_name = case_name_raw.replace("Copy of ", "")
        
        if case_name not in target_leads:
            continue
            
        print(f"Processing {case_name}...")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        res = remove_grid(img, {})
        
        ink = res["ink_candidate"]
        grid = res["grid_candidate"]
        glob_sep = res["global_separators"]
        protected = res["waveform_protected"]
        
        # We need to compute the requested debug masks
        # 01_input.png = img
        # 02_grid_horizontal_candidates = grid_h (we will approximate it if not in res, wait, I can modify m6 to return it, or just use what we have)
        # Actually I didn't return grid_h or grid_v in m6. I will just recompute them here for visualization.
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink_candidate = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        
        k_y = max(5, int(h * 0.05))
        k_x = max(5, int(w * 0.05))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_y))
        grid_h = cv2.morphologyEx(ink_candidate, cv2.MORPH_OPEN, horiz_kernel)
        grid_v = cv2.morphologyEx(ink_candidate, cv2.MORPH_OPEN, vert_kernel)
        
        # Let's crop to the approximate lead regions based on standard 3x4 layout to satisfy the user's lead-specific output
        row_h = h // 4
        col_w = w // 4
        
        lead_coords = {
            "I": (0,0), "II": (1,0), "III": (2,0),
            "aVR": (0,1), "aVL": (1,1), "aVF": (2,1),
            "V1": (0,2), "V2": (1,2), "V3": (2,2),
            "V4": (0,3), "V5": (1,3), "V6": (2,3)
        }
        
        for lead in target_leads[case_name]:
            r, c = lead_coords[lead]
            
            ymin = r * row_h
            ymax = (r+1) * row_h
            xmin = c * col_w
            xmax = (c+1) * col_w
            
            # extract crops
            c_input = img[ymin:ymax, xmin:xmax]
            c_grid_h = grid_h[ymin:ymax, xmin:xmax]
            c_grid_v = grid_v[ymin:ymax, xmin:xmax]
            c_glob_sep = glob_sep[ymin:ymax, xmin:xmax]
            c_conf_sep = glob_sep[ymin:ymax, xmin:xmax] # Same since we did hard exclusion
            c_protected = protected[ymin:ymax, xmin:xmax]
            
            # 07_final_grid_removed is just the protected mask
            c_final = protected[ymin:ymax, xmin:xmax]
            
            # 08_overlay_removed
            c_overlay = c_input.copy()
            c_overlay[c_glob_sep > 0] = (0, 0, 255) # Red removed separators
            
            # 09_before_after_stage3
            # We can visualize the previous approach (no hard exclusion) vs the new one
            # previous approach:
            c_ink = ink[ymin:ymax, xmin:xmax]
            c_grid = grid[ymin:ymax, xmin:xmax]
            wf_init = cv2.bitwise_and(c_ink, cv2.bitwise_not(c_grid))
            bridge_k = max(2, int(min(h, w) * 0.015))
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_k, bridge_k))
            exp_init = cv2.dilate(wf_init, bridge_kernel, iterations=1)
            prev_protected = cv2.bitwise_and(c_ink, exp_init)
            
            # create side by side
            prev_color = cv2.cvtColor(prev_protected, cv2.COLOR_GRAY2BGR)
            new_color = cv2.cvtColor(c_final, cv2.COLOR_GRAY2BGR)
            
            # draw red circles over differences
            diff = cv2.absdiff(prev_protected, c_final)
            new_color[diff > 0] = (0, 0, 255) # Highlight removed pixels in red on the right
            
            combined = np.hstack((prev_color, new_color))
            
            # 10_separator_removal_overlay
            sep_overlay = c_input.copy()
            sep_overlay[c_glob_sep > 0] = (0, 255, 255) # Yellow removed
            
            lead_dir = os.path.join(out_base, f"{case_name}_{lead}")
            os.makedirs(lead_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(lead_dir, "01_input.png"), c_input)
            cv2.imwrite(os.path.join(lead_dir, "02_grid_horizontal_candidates.png"), c_grid_h)
            cv2.imwrite(os.path.join(lead_dir, "03_grid_vertical_candidates.png"), c_grid_v)
            cv2.imwrite(os.path.join(lead_dir, "04_global_separator_candidates.png"), c_glob_sep)
            cv2.imwrite(os.path.join(lead_dir, "05_confirmed_separator_mask.png"), c_conf_sep)
            cv2.imwrite(os.path.join(lead_dir, "06_waveform_protected.png"), prev_protected)
            cv2.imwrite(os.path.join(lead_dir, "07_final_grid_removed.png"), c_final)
            cv2.imwrite(os.path.join(lead_dir, "08_overlay_removed.png"), c_overlay)
            cv2.imwrite(os.path.join(lead_dir, "09_before_after_stage3.png"), combined)
            cv2.imwrite(os.path.join(lead_dir, "10_separator_removal_overlay.png"), sep_overlay)
            
            print(f"Generated outputs for {case_name} {lead}")

if __name__ == "__main__":
    run()
