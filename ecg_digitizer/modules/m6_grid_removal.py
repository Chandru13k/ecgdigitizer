import cv2
import numpy as np

def remove_grid(color_patch, config):
    h, w = color_patch.shape[:2]
    gray = cv2.cvtColor(color_patch, cv2.COLOR_BGR2GRAY)
    
    # 1. Ink Candidate: Adaptive threshold for ALL dark printed matter
    ink_candidate = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    
    # 2. Adaptive Grid Scale Estimation
    k_y = max(5, int(h * 0.05))
    k_x = max(5, int(w * 0.05))
    
    # 3. Grid Candidate: Find long horizontal and vertical lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_y))
    
    grid_h = cv2.morphologyEx(ink_candidate, cv2.MORPH_OPEN, horiz_kernel)
    grid_v = cv2.morphologyEx(ink_candidate, cv2.MORPH_OPEN, vert_kernel)
    grid_candidate = cv2.bitwise_or(grid_h, grid_v)
    
    # 3.5 Global Separator Detection (NEW)
    # Detect persistent vertical structures spanning >40% of the image height
    sep_k_y = max(10, int(h * 0.40))
    global_vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, sep_k_y))
    global_separators = cv2.morphologyEx(ink_candidate, cv2.MORPH_OPEN, global_vert_kernel)
    
    # Optional: Slightly dilate global separators to ensure clean removal of their edges
    sep_dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    global_separators = cv2.dilate(global_separators, sep_dilate_k, iterations=1)
    
    # 4. Waveform Initial: Ink that is definitely not grid
    waveform_initial = cv2.bitwise_and(ink_candidate, cv2.bitwise_not(grid_candidate))
    
    # 5. Protected Waveform: Preserve grid intersections where waveform physically crosses
    bridge_k = max(2, int(min(h, w) * 0.015))
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_k, bridge_k))
    expanded_initial = cv2.dilate(waveform_initial, bridge_kernel, iterations=1)
    
    waveform_protected = cv2.bitwise_and(ink_candidate, expanded_initial)
    
    # 6. HARD EXCLUSION: Remove confirmed global separators so they are not resurrected
    waveform_protected = cv2.bitwise_and(waveform_protected, cv2.bitwise_not(global_separators))
    
    return {
        "grid_candidate": grid_candidate,
        "ink_candidate": ink_candidate,
        "global_separators": global_separators,
        "waveform_initial": waveform_initial,
        "waveform_protected": waveform_protected
    }
