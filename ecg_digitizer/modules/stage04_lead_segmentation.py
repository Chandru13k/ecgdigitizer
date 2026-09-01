import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

LEAD_MATRIX = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

def find_minimum_energy_valley(fg_mask, y_search_start, y_search_end, x_start, x_end):
    """
    Computes a 1D horizontal projection within the specified search window and column,
    smooths it, and returns the Y-coordinate of the absolute minimum energy.
    """
    if y_search_end <= y_search_start:
        return y_search_start
        
    region = fg_mask[y_search_start:y_search_end, x_start:x_end]
    projection = np.sum(region > 0, axis=1)
    smoothed = gaussian_filter1d(projection, sigma=2.0)
    
    # Find absolute minimum
    min_idx = np.argmin(smoothed)
    return y_search_start + min_idx

def run_stage4(image, stage3_metrics, stage2_2_metrics):
    """
    Stage 4: Geometric Lead ROI Localization using Minimum-Energy Valleys.
    Outputs strictly non-overlapping bounding boxes covering the entire grid.
    Modifies NO pixels.
    """
    h, w = image.shape[:2]
    
    # 1. Parse Stage 3 Column Boundaries
    raw_lines = sorted([vl["x_coordinate"] for vl in stage3_metrics.get("vertical_lines_details", [])])
    if len(raw_lines) == 3:
        raw_lines.insert(0, 0)
        
    if len(raw_lines) < 4:
        raise ValueError(f"Expected at least 4 vertical separators, found {len(raw_lines)}")
        
    # Standard 12-lead grid layouts have 4 major columns. If there are extra lines (like text boxes/markers),
    # select the 4 coordinates closest to standard grid division columns (0%, 25%, 50%, 75% of width).
    vert_lines = []
    target_fractions = [0.0, 0.25, 0.50, 0.75]
    for frac in target_fractions:
        target_x = int(frac * w)
        closest_line = min(raw_lines, key=lambda val: abs(val - target_x))
        vert_lines.append(closest_line)
    vert_lines = sorted(list(set(vert_lines)))
    
    # Pad to make sure we have exactly 4 separators if deduplication removed one
    while len(vert_lines) < 4:
        for val in raw_lines:
            if val not in vert_lines:
                vert_lines.append(val)
                break
        vert_lines = sorted(vert_lines)
        
    cols_x = []
    for i in range(len(vert_lines)):
        x_start = vert_lines[i]
        x_end = vert_lines[i+1] if i+1 < len(vert_lines) else w
        cols_x.append({"x1": max(0, x_start), "x2": min(w, x_end)})
        
    # 2. Parse Stage 2 Row Boundaries as Priors
    full_img_h = stage3_metrics.get("input_height", 2481)
    active_y1 = stage2_2_metrics.get("active_waveform_y1", 0)
    
    detected_rows = stage2_2_metrics.get("detected_rows", [])
    if len(detected_rows) < 4:
        # Fallback to standard standard layout grid rows (4 rows divided equally across active height)
        row_priors = []
        dy = h // 4
        for r_idx in range(4):
            row_priors.append({"y1": r_idx * dy, "y2": min(h, (r_idx + 1) * dy)})
    else:
        row_priors = []
        for row in detected_rows:
            r_y1 = int(row["normalized_y1"] * full_img_h) - active_y1
            r_y2 = int(row["normalized_y2"] * full_img_h) - active_y1
            row_priors.append({"y1": max(0, r_y1), "y2": min(h, r_y2)})
        
    # 3. Find Minimum-Energy Valleys per Column
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, fg_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    rois = {}
    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
    for c_idx, col in enumerate(cols_x):
        cx1, cx2 = col["x1"], col["x2"]
        
        # Calculate the 3 valley boundaries for this column
        # Valley 1: between Row 0 and Row 1
        v1_search_start = row_priors[0]["y2"]
        v1_search_end = row_priors[1]["y1"]
        valley1 = find_minimum_energy_valley(fg_mask, v1_search_start - 20, v1_search_end + 20, cx1, cx2)
        
        # Valley 2: between Row 1 and Row 2
        v2_search_start = row_priors[1]["y2"]
        v2_search_end = row_priors[2]["y1"]
        valley2 = find_minimum_energy_valley(fg_mask, v2_search_start - 20, v2_search_end + 20, cx1, cx2)
        
        # Valley 3: between Row 2 and Rhythm II (Row 3)
        v3_search_start = row_priors[2]["y2"]
        v3_search_end = row_priors[3]["y1"]
        valley3 = find_minimum_energy_valley(fg_mask, v3_search_start - 20, v3_search_end + 20, cx1, cx2)
        
        # Assign bounds for the 3 short leads in this column
        # Row 0
        y1, y2 = 0, valley1
        lead_name = LEAD_MATRIX[0][c_idx]
        rois[lead_name] = _create_roi_dict(lead_name, 0, c_idx, cx1, cx2, y1, y2)
        _draw_debug(debug_img, lead_name, cx1, cx2, y1, y2, (255, 0, 0))
        
        # Row 1
        y1, y2 = valley1, valley2
        lead_name = LEAD_MATRIX[1][c_idx]
        rois[lead_name] = _create_roi_dict(lead_name, 1, c_idx, cx1, cx2, y1, y2)
        _draw_debug(debug_img, lead_name, cx1, cx2, y1, y2, (0, 200, 0))
        
        # Row 2
        y1, y2 = valley2, valley3
        lead_name = LEAD_MATRIX[2][c_idx]
        rois[lead_name] = _create_roi_dict(lead_name, 2, c_idx, cx1, cx2, y1, y2)
        _draw_debug(debug_img, lead_name, cx1, cx2, y1, y2, (200, 0, 200))

    # 4. Rhythm II (Global Valley 3 to Bottom)
    # Rhythm spans full width. We'll find a global valley between Row 2 and Rhythm II
    v3_search_start = row_priors[2]["y2"]
    v3_search_end = row_priors[3]["y1"]
    rhythm_valley = find_minimum_energy_valley(fg_mask, v3_search_start - 20, v3_search_end + 20, 0, w)
    
    rois["Rhythm_II"] = _create_roi_dict("Rhythm_II", 3, 0, 0, w, rhythm_valley, h)
    _draw_debug(debug_img, "Rhythm_II", 0, w, rhythm_valley, h, (0, 0, 255))
    
    metrics = {
        "stage": "4",
        "output_width": w,
        "output_height": h,
        "total_leads_detected": len(rois)
    }
    
    return rois, debug_img, metrics

def _create_roi_dict(lead_name, row, col, x1, x2, y1, y2):
    return {
        "lead_name": lead_name,
        "row": row,
        "column": col,
        "x_start": int(x1),
        "x_end": int(x2),
        "y_start": int(y1),
        "y_end": int(y2),
        "baseline_y": None,
        "baseline_source": "not_estimated_in_stage4"
    }

def _draw_debug(img, label, x1, x2, y1, y2, color):
    # Draw inset rectangle to not overlap pixel boundaries exactly
    cv2.rectangle(img, (x1+1, y1+1), (x2-1, y2-1), color, 2)
    cv2.putText(img, label, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
