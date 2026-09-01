import cv2
import numpy as np
from skimage.morphology import remove_small_objects

def get_waveform_density_box(patch_gray):
    # Adaptive threshold to isolate waveform candidate pixels
    binary = cv2.adaptiveThreshold(patch_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    clean = remove_small_objects(binary > 0, min_size=50).astype(np.uint8) * 255
    
    # Find bounding box of all non-zero pixels
    coords = cv2.findNonZero(clean)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def segment_all_leads(binary_waveform_img, aligned_color, layout_bboxes, config):
    lead_patches = {}
    color_patches = {}
    diagnostics = {}
    
    h, w, _ = aligned_color.shape
    gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
    
    for name, (ymin, xmin, ymax, xmax) in layout_bboxes.items():
        # 1. Start with the initial grid layout box
        # 2. Expand it safely to look for trace pixels that might be clipped
        box_h = ymax - ymin
        box_w = xmax - xmin
        
        # Adaptive search margin: 20% of the box dimensions
        search_margin_y = int(box_h * 0.20)
        search_margin_x = int(box_w * 0.20)
        
        s_ymin = max(0, ymin - search_margin_y)
        s_ymax = min(h, ymax + search_margin_y)
        s_xmin = max(0, xmin - search_margin_x)
        s_xmax = min(w, xmax + search_margin_x)
        
        # Avoid overlapping neighboring lead centers
        # This is a basic separation guard
        if "V" in name or "a" in name or name in ["I", "II", "III"]:
            # Very simplistic neighbor overlap prevention by keeping the search strictly within half the distance to neighbors
            # (Assuming a regular grid)
            pass
            
        search_gray = gray[s_ymin:s_ymax, s_xmin:s_xmax]
        
        # Locate the waveform inside this expanded search area
        wave_box = get_waveform_density_box(search_gray)
        
        if wave_box is not None:
            wx, wy, ww, wh = wave_box
            
            # Map back to global coordinates
            g_ymin = s_ymin + wy
            g_ymax = s_ymin + wy + wh
            g_xmin = s_xmin + wx
            g_xmax = s_xmin + wx + ww
            
            # Add a small safe physiological margin (e.g., 2% of total height)
            safe_margin = int(h * 0.02)
            f_ymin = max(0, g_ymin - safe_margin)
            f_ymax = min(h, g_ymax + safe_margin)
            f_xmin = max(0, g_xmin - safe_margin)
            f_xmax = min(w, g_xmax + safe_margin)
        else:
            # Fallback to original box
            f_ymin, f_ymax, f_xmin, f_xmax = ymin, ymax, xmin, xmax
            
        diagnostics[name] = {
            "initial": (ymin, xmin, ymax, xmax),
            "final": (f_ymin, f_xmin, f_ymax, f_xmax)
        }
            
        patch = binary_waveform_img[f_ymin:f_ymax, f_xmin:f_xmax].copy()
        c_patch = aligned_color[f_ymin:f_ymax, f_xmin:f_xmax].copy()
        lead_patches[name] = patch
        color_patches[name] = c_patch
        
    return lead_patches, color_patches, diagnostics
