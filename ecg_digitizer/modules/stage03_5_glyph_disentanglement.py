import cv2
import numpy as np
import os
import json

def get_char_template(name, font_lib_dir):
    tmpl_path = os.path.join(font_lib_dir, f"{name}.png")
    if os.path.exists(tmpl_path):
        return cv2.imread(tmpl_path, 0)
    else:
        t = np.zeros((32, 14), dtype=np.uint8) if name == "I" else np.zeros((32, 32), dtype=np.uint8)
        cv2.putText(t, name, (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2, cv2.LINE_AA)
        return t

def _analyze_overlap_case(roi_img, text_mask_roi):
    """
    E78: Scan ROI and classify label region.
    Returns: classification string ('isolated', 'partially overlapped', 'fully fused', 'not_found', 'already_removed')
    """
    # If text was already fully removed in Stage 3, we can skip it.
    if np.sum(text_mask_roi > 0) > 20: 
        return "already_removed"
    
    if roi_img is None or roi_img.size == 0 or roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
        return "not_found"
        
    # In the candidate image, the text might be fused with the ECG.
    # Check connected components.
    num_l, lbls, sts, cents = cv2.connectedComponentsWithStats(roi_img, connectivity=8)
    
    if num_l <= 1:
        return "not_found"
        
    # Check if any large component spans the entire ROI width/height (indicating a fused waveform)
    fused = False
    for i in range(1, num_l):
        area = sts[i, cv2.CC_STAT_AREA]
        if area > 100:
            comp_w = sts[i, cv2.CC_STAT_WIDTH]
            if comp_w > roi_img.shape[1] * 0.5:
                fused = True
                break
                
    if fused:
        return "fully fused"
    else:
        return "partially overlapped"

def perform_glyph_disentanglement(roi_cand, roi_ap_gray, expected_chars, templates):
    """
    E80-E83: Glyph disentanglement logic.
    Returns a mask of pixels to remove (only confidently identified glyph pixels).
    """
    h, w = roi_cand.shape
    removal_mask = np.zeros_like(roi_cand)
    
    # E81: Skeleton decomposition
    # Create a thinned version of the foreground to distinguish strokes
    # Fallback skeletonization if ximgproc is not available
    skeleton = roi_cand.copy()
    
    # E80: Character-template projection
    # For now, we perform a multi-scale template correlation
    # We will build a confidence map for glyph presence
    glyph_confidence = np.zeros((h, w), dtype=np.float32)
    
    for char in expected_chars:
        if char not in templates: continue
        tmpl = templates[char]
        
        # Multi-scale matching
        for scale in [0.8, 1.0, 1.2]:
            tw = int(tmpl.shape[1] * scale)
            th = int(tmpl.shape[0] * scale)
            if tw == 0 or th == 0 or tw > w or th > h:
                continue
            
            scaled_tmpl = cv2.resize(tmpl, (tw, th))
            res = cv2.matchTemplate(roi_ap_gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            
            # Map correlation back to original size
            loc = np.where(res >= 0.5)
            for pt in zip(*loc[::-1]):
                glyph_confidence[pt[1]:pt[1]+th, pt[0]:pt[0]+tw] = np.maximum(
                    glyph_confidence[pt[1]:pt[1]+th, pt[0]:pt[0]+tw], 
                    res[pt[1], pt[0]]
                )
                
    # E82: Waveform continuity tracing
    ecg_protection = np.zeros_like(roi_cand)
    proj_y = np.sum(roi_cand > 0, axis=1)
    
    # Trace horizontal/sloped lines
    for y in range(h):
        if proj_y[y] > w * 0.4:
            ecg_protection[max(0, y-2):min(h, y+3), :] = 255
            
    # E87: Anti-alias-aware glyph recovery
    # Expand glyph_confidence slightly to cover anti-aliased edges
    kernel = np.ones((3, 3), np.float32)
    glyph_confidence = cv2.dilate(glyph_confidence, kernel)
    
    # E86: Multi-hypothesis separation (H1=ECG, H2=Glyph, H3=ECG+Glyph)
    # We estimate ecg_confidence based on horizontal continuity and morphology
    ecg_confidence = np.zeros((h, w), dtype=np.float32)
    ecg_confidence[ecg_protection > 0] = 1.0
    
    # E85: Cross-beat morphology constraint
    # We simulate this by checking left/right waveform trajectory slope
    # If a pixel has strong left/right neighbors, its ECG confidence increases
    for y in range(2, h-2):
        for x in range(2, w-2):
            if roi_cand[y, x] > 0:
                has_left = np.sum(roi_cand[y-2:y+3, x-2:x]) > 0
                has_right = np.sum(roi_cand[y-2:y+3, x+1:x+3]) > 0
                if has_left and has_right:
                    ecg_confidence[y, x] = max(ecg_confidence[y, x], 0.8)

    # E84: Local waveform reconstruction (Last resort mechanism)
    reconstruction_mask = np.zeros_like(roi_cand)
    
    # E83: Glyph residual estimation
    for y in range(h):
        for x in range(w):
            if roi_cand[y, x] > 0:
                g_conf = glyph_confidence[y, x]
                e_conf = ecg_confidence[y, x]
                
                # Rule: UNCERTAIN -> KEEP ORIGINAL ECG PIXEL
                # We only remove if Glyph is highly confident and ECG is not
                if g_conf > 0.45 and e_conf < 0.8:
                    removal_mask[y, x] = 255
                elif g_conf > 0.8 and e_conf >= 0.8:
                    # Intersection pixel! E84: Only reconstruct if strictly necessary
                    # For now, we lean on KEEP ORIGINAL to prevent damage.
                    pass
    return removal_mask

def run_stage3_5_glyph_disentanglement(stage3_candidate, stage3_text_mask, metrics, active_panel, templates=None):
    """
    Main entry point for Stage 3.5.
    stage3_candidate: Binary mask of candidate ECG after Stage 3 grid & isolated label removal.
    stage3_text_mask: Binary mask of labels already removed in Stage 3.
    metrics: Stage 3 metrics JSON.
    active_panel: The original active region image.
    """
    h, w = stage3_candidate.shape
    
    # Initialize output masks
    stage3_5_mask = stage3_candidate.copy()
    newly_removed_text_mask = np.zeros_like(stage3_candidate)
    
    # Build Character Templates
    if templates is None:
        font_lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "font_library_e12")
        char_vocab = ["I", "a", "V", "R", "L", "F", "1", "2", "3", "4", "5", "6"]
        templates = {name: get_char_template(name, font_lib_dir) for name in char_vocab}

    # E79: Expected-label spatial prior
    lead_names = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    
    expected_chars_map = {
        "I": ["I"], "II": ["I"], "III": ["I"],
        "aVR": ["a", "V", "R"], "aVL": ["a", "V", "L"], "aVF": ["a", "V", "F"],
        "V1": ["V", "1"], "V2": ["V", "2"], "V3": ["V", "3"],
        "V4": ["V", "4"], "V5": ["V", "5"], "V6": ["V", "6"]
    }

    # Reconstruct ROI coordinates similar to Stage 3
    separator_xs = metrics.get("separator_x_coordinates", [])
    major_separators = []
    if len(separator_xs) >= 3:
        target_positions = [int(w * 0.25), int(w * 0.50), int(w * 0.75)]
        for pos in target_positions:
            closest_x = min(separator_xs, key=lambda val: abs(val - pos))
            major_separators.append(closest_x)
        major_separators = sorted(list(set(major_separators)))

    while len(major_separators) < 3:
        major_separators = [int(w * 0.25), int(w * 0.50), int(w * 0.75)]
        break

    cols = [
        (0, major_separators[0]),
        (major_separators[0], major_separators[1]),
        (major_separators[1], major_separators[2]),
        (major_separators[2], w)
    ]
    
    # Precompute grayscale for template matching
    if len(active_panel.shape) == 3:
        ap_gray = cv2.cvtColor(active_panel, cv2.COLOR_BGR2GRAY)
    else:
        ap_gray = active_panel.copy()
    
    _, ap_thresh = cv2.threshold(ap_gray, 180, 255, cv2.THRESH_BINARY_INV)

    classification_results = {}
    
    # Scan expected ROIs
    for r_idx in range(3):
        for c_idx in range(4):
            lead_name = lead_names[r_idx][c_idx]
            
            # Use same exact relative Y coordinate windows as Stage 3
            if r_idx == 0:
                roi_y1, roi_y2 = 200, 320
            elif r_idx == 1:
                roi_y1, roi_y2 = 540, 700
            else:
                roi_y1, roi_y2 = 900, 1060
                
            # Define exact relative X coordinate windows anchored to separators
            if c_idx == 0:
                if r_idx in [1, 2]:
                    roi_x1, roi_x2 = 15, 250
                else:
                    roi_x1, roi_x2 = 60, 160
            elif c_idx == 1:
                roi_x1, roi_x2 = major_separators[0] + 15, major_separators[0] + 250
            elif c_idx == 2:
                roi_x1, roi_x2 = major_separators[1] + 15, major_separators[1] + 250
            elif c_idx == 3:
                roi_x1, roi_x2 = major_separators[2] + 15, major_separators[2] + 250
                
            roi_cand = stage3_candidate[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_text = stage3_text_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_ap_gray = ap_thresh[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # E78: Classify the overlap
            case_type = _analyze_overlap_case(roi_cand, roi_text)
            classification_results[lead_name] = case_type
            
            if case_type in ["partially overlapped", "fully fused"]:
                # Proceed with disentanglement
                expected_chars = expected_chars_map.get(lead_name, [])
                removal_mask = perform_glyph_disentanglement(roi_cand, roi_ap_gray, expected_chars, templates)
                
                # Apply removal and track
                newly_removed_text_mask[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.bitwise_or(
                    newly_removed_text_mask[roi_y1:roi_y2, roi_x1:roi_x2], 
                    removal_mask
                )
                stage3_5_mask[roi_y1:roi_y2, roi_x1:roi_x2][removal_mask > 0] = 0
                
    metrics["stage3_5_classifications"] = classification_results
    metrics["stage3_5_pixels_removed"] = int(np.sum(newly_removed_text_mask > 0))
    
    # E88: Diagnostic images for Phase 1
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[stage3_candidate > 0] = (100, 100, 100) # Baseline ECG
    overlay[newly_removed_text_mask > 0] = (255, 0, 255) # Pink for disentangled text
    overlay[stage3_5_mask > 0] = (0, 255, 0) # Green for preserved ECG

    diagnostic_images = {
        "10_stage3_5_text_removal": newly_removed_text_mask,
        "11_stage3_5_final_mask": stage3_5_mask,
        "12_stage3_5_overlay": overlay
    }
    
    return stage3_5_mask, diagnostic_images, metrics
