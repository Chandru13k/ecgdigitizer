import cv2
import numpy as np
import hashlib

def refine_ecg_panel(image, metrics):
    """
    Stage 2.2: Precision Refinement
    Validates input integrity and directionally refines the Stage 2.1 bounding box.
    """
    if metrics.get("stage") != "stage2_1":
        raise ValueError("PIPELINE INTEGRITY FAILURE: Input metrics must be from Stage 2.1")
        
    actual_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    if metrics.get("source_sha256") != actual_sha256:
        raise ValueError("PIPELINE INTEGRITY FAILURE: Input image hash does not match Stage 2.1 source_sha256")
        
    h, w = image.shape[:2]
    
    bbox = metrics["ecg_panel_bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    if not (0 <= x1 < x2 <= w) or not (0 <= y1 < y2 <= h):
        raise ValueError("PIPELINE INTEGRITY FAILURE: Invalid Stage 2.1 coordinates")
        
    initial_x1, initial_y1, initial_x2, initial_y2 = x1, y1, x2, y2
    
    # 1. Estimate Grid Spacing using a robust bootstrap morphology
    # We use a very small kernel relative to image dimensions just to kill noise and small text.
    # The actual structural detection will use kernels based on the detected grid spacing.
    initial_k_horiz = max(5, int(w * 0.005))
    initial_k_vert = max(5, int(h * 0.005))
    kernel_h_init = cv2.getStructuringElement(cv2.MORPH_RECT, (initial_k_horiz, 1))
    kernel_v_init = cv2.getStructuringElement(cv2.MORPH_RECT, (1, initial_k_vert))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    _, binary_grad = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
    
    lines_h_init = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_h_init)
    lines_v_init = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_v_init)
    initial_grid = cv2.bitwise_or(lines_h_init, lines_v_init)
    
    # Compute projections on the cleaned initial grid to estimate grid spacing robustly
    raw_h_proj = np.sum(initial_grid, axis=1) / 255.0
    raw_v_proj = np.sum(initial_grid, axis=0) / 255.0
    
    def estimate_spacing(proj):
        # Smooth the projection to merge thick lines into single peaks
        proj_smooth = np.convolve(proj, np.ones(5)/5.0, mode='same')
        peaks = []
        for i in range(1, len(proj_smooth)-1):
            if proj_smooth[i] > proj_smooth[i-1] and proj_smooth[i] > proj_smooth[i+1] and proj_smooth[i] > np.max(proj_smooth)*0.1:
                peaks.append(i)
        
        # Filter out peaks that are too close (merge them)
        merged_peaks = []
        for p in peaks:
            if not merged_peaks or p - merged_peaks[-1] > 10:
                merged_peaks.append(p)
                
        if len(merged_peaks) > 1:
            diffs = np.diff(merged_peaks)
            return np.median(diffs)
        return max(5.0, min(h, w) * 0.01) # fallback
        
    grid_spacing_y = estimate_spacing(raw_h_proj)
    grid_spacing_x = estimate_spacing(raw_v_proj)
    
    # 2. Extract Structural Grid from Full Image using Detected Spacing
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    _, binary_grad = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
    
    # Use structurally derived morphology kernels
    # True grid lines span multiple grid cells. By requiring lines to be at least 2 grid cells long,
    # we mathematically destroy small text characters and isolated noise.
    k_horiz = max(3, int(grid_spacing_x * 2.5))
    k_vert = max(3, int(grid_spacing_y * 2.5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_horiz, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
    
    lines_h = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_v)
    grid_structure = cv2.bitwise_or(lines_h, lines_v)
    
    h_proj = np.sum(grid_structure, axis=1) / 255.0  # length h
    v_proj = np.sum(grid_structure, axis=0) / 255.0  # length w
    
    # 3. Waveform classification
    _, ink_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    
    def is_waveform_or_calibration(w, h, area, side):
        if area < 50: return False # noise
        aspect = w / float(h + 1e-5)
        fill_ratio = area / (w * h + 1e-5)
        
        # Calibration pulse: blocky, but not the whole page!
        if 400 < area < 50000 and aspect < 1.5 and h > grid_spacing_y and h < h_proj.shape[0] * 0.2:
            return True
            
        # Large vertical excursions (QRS spikes): tall, narrow, low fill ratio
        if h > grid_spacing_y * 1.2 and w < grid_spacing_x * 2.0 and fill_ratio < 0.4 and h < h_proj.shape[0] * 0.3:
            return True
            
        # If checking the bottom boundary, we must be extremely strict to avoid report text.
        if side == "bottom":
            return False # Only calibration pulses or QRS spikes should cross the bottom boundary.
            
        # Waveform line: wide, low fill ratio (useful for left/right boundaries)
        if w > grid_spacing_x * 1.5 and fill_ratio < 0.4 and h < h_proj.shape[0] * 0.5 and w < v_proj.shape[0] * 0.9:
            return True
            
        # For other sides, be slightly more forgiving for generic large components
        if side != "bottom" and area > 1000 and fill_ratio < 0.6 and h < h_proj.shape[0] * 0.9 and w < v_proj.shape[0] * 0.9:
            return True
            
        return False

    # 3. Level A - Structural Grid Bounds
    # Trace the structural bounds using the projections from the center outwards
    mid_y = y1 + (y2 - y1) // 2
    mid_x = x1 + (x2 - x1) // 2
    
    h_thresh = np.max(h_proj) * 0.05
    v_thresh = np.max(v_proj) * 0.05
    
    grid_y2 = min(h - 1, y2)
    while grid_y2 > mid_y and h_proj[grid_y2] < h_thresh: grid_y2 -= 1
    
    grid_y1 = max(0, y1)
    while grid_y1 < mid_y and h_proj[grid_y1] < h_thresh: grid_y1 += 1
    
    grid_x2 = min(w - 1, x2)
    while grid_x2 > mid_x and v_proj[grid_x2] < v_thresh: grid_x2 -= 1
    
    grid_x1 = max(0, x1)
    while grid_x1 < mid_x and v_proj[grid_x1] < v_thresh: grid_x1 += 1
    
    # 4. Level B - Active ECG Panel Detection
    # Isolate pure dark ink to ignore the pink grid lines completely
    _, pure_ink_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Extract connected components from the pure ink mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pure_ink_mask, connectivity=8)
    
    valid_bboxes = []
    calibration_bboxes = []
    
    for lbl in range(1, num_labels):
        cx = stats[lbl, cv2.CC_STAT_LEFT]
        cy = stats[lbl, cv2.CC_STAT_TOP]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        carea = stats[lbl, cv2.CC_STAT_AREA]
        
        # Limit analysis strictly to the detected structural grid region
        if cy < grid_y1 or cy + ch > grid_y2 or cx < grid_x1 or cx + cw > grid_x2:
            continue
            
        fill_ratio = carea / (cw * ch + 1e-5)
        
        # Calibration discrimination
        # Calibration is typically strongly rectangular, has a higher fill ratio than thin waveforms,
        # and has a localized horizontal occupancy.
        # Extreme QRS excursions are very tall but have extremely low fill ratio (e.g. < 0.03).
        is_cal = cw < grid_spacing_x * 5.0 and ch > grid_spacing_y * 3.0 and fill_ratio > 0.05
        
        if is_cal:
            calibration_bboxes.append((cx, cy, cw, ch))
            continue # Reject from active waveforms
        
        # Identify genuine waveform traces and QRS excursions using structural dimensions
        is_trace = cw > grid_spacing_x * 3.0 and fill_ratio < 0.4
        is_qrs = ch > grid_spacing_y * 2.0 and cw < grid_spacing_x * 2.5 and fill_ratio < 0.35
        is_complex = cw > grid_spacing_x * 2.5 and ch > grid_spacing_y * 1.5 and fill_ratio < 0.35
        
        if is_trace or is_qrs or is_complex:
            valid_bboxes.append((cx, cy, cw, ch))
            
    # Cluster valid waveform components by y-center to discover distinct ECG rows
    row_centers = []
    for cx, cy, cw, ch in valid_bboxes:
        mid_cy = cy + ch // 2
        found = False
        for i, (rc, bboxes) in enumerate(row_centers):
            # If the component falls within a 3-grid-cell vertical radius of a row center, cluster it
            if abs(rc - mid_cy) < grid_spacing_y * 3:
                row_centers[i][1].append((cx, cy, cw, ch))
                found = True
                break
        if not found:
            row_centers.append((mid_cy, [(cx, cy, cw, ch)]))
            
    row_centers.sort(key=lambda x: x[0])
    
    # Identify the primary (genuine) ECG rows that span the majority of the panel width
    genuine_rows = []
    structural_width = grid_x2 - grid_x1
    for rc, bboxes in row_centers:
        min_x = min(b[0] for b in bboxes)
        max_x = max(b[0] + b[2] for b in bboxes)
        total_w = max_x - min_x
        
        # A primary ECG row must span at least 60% of the structural grid width
        if total_w > structural_width * 0.6: 
            genuine_rows.append((rc, bboxes))
            
    if not genuine_rows:
        # Fallback to structural grid if no rows detected
        final_y1, final_y2 = grid_y1, grid_y2
        final_x1, final_x2 = grid_x1, grid_x2
    else:
        # Compute Active ECG Panel bounds by taking the union of all genuine rows
        row_min_y = min(min(b[1] for b in bboxes) for _, bboxes in genuine_rows)
        row_max_y = max(max(b[1] + b[3] for b in bboxes) for _, bboxes in genuine_rows)
        row_min_x = min(min(b[0] for b in bboxes) for _, bboxes in genuine_rows)
        row_max_x = max(max(b[0] + b[2] for b in bboxes) for _, bboxes in genuine_rows)
        
        # Apply a structural safety margin based on grid spacing, bounded by structural grid
        margin_y = int(grid_spacing_y * 1.5)
        margin_x = int(grid_spacing_x * 1.5)
        
        final_y1 = max(grid_y1, row_min_y - margin_y)
        final_y2 = min(grid_y2, row_max_y + margin_y)
        
        # Use structural grid for X bounds as requested
        final_x1 = grid_x1
        final_x2 = grid_x2
        
    x1, y1, x2, y2 = final_x1, final_y1, final_x2, final_y2

    def status(initial, current):
        if current > initial: return "tightened" if initial == initial_x1 or initial == initial_y1 else "expanded"
        if current < initial: return "expanded" if initial == initial_x1 or initial == initial_y1 else "tightened"
        return "preserved"
        
    debug_images = {
        "grid_evidence": grid_structure
    }
    
    if len(image.shape) == 3:
        # Visualize Row Detection
        vis_rows = image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
        for i, (rc, bboxes) in enumerate(genuine_rows):
            c = colors[i % len(colors)]
            for bx, by, bw, bh in bboxes:
                cv2.rectangle(vis_rows, (bx, by), (bx+bw, by+bh), c, 2)
            # Draw row center
            cv2.line(vis_rows, (final_x1, int(rc)), (final_x2, int(rc)), c, 2)
        debug_images["04_row_detection"] = vis_rows
        
        # Visualize Calibration Rejection
        vis_cal = image.copy()
        # Draw all waveforms in green
        for i, (rc, bboxes) in enumerate(genuine_rows):
            for bx, by, bw, bh in bboxes:
                cv2.rectangle(vis_cal, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
        # Draw rejected calibration pulses in red
        for cx, cy, cw, ch in calibration_bboxes:
            cv2.rectangle(vis_cal, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 4)
        cv2.rectangle(vis_cal, (final_x1, final_y1), (final_x2, final_y2), (255, 0, 0), 4)
        debug_images["05_calibration_rejection"] = vis_cal

    detected_rows_out = []
    for rc, bboxes in genuine_rows:
        if not bboxes:
            continue
        r_min_y = min(b[1] for b in bboxes)
        r_max_y = max(b[1] + b[3] for b in bboxes)
        r_min_x = min(b[0] for b in bboxes)
        r_max_x = max(b[0] + b[2] for b in bboxes)
        detected_rows_out.append({
            "normalized_y1": float(r_min_y) / h,
            "normalized_y2": float(r_max_y) / h,
            "normalized_x1": float(r_min_x) / w,
            "normalized_x2": float(r_max_x) / w,
            "component_count": len(bboxes)
        })

    return {
        "stage": "stage2_2",
        "source_sha256": actual_sha256,
        "stage2_1_bbox": {
            "x1": initial_x1,
            "y1": initial_y1,
            "x2": initial_x2,
            "y2": initial_y2
        },
        "stage2_2_bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        },
        "normalized_bbox": {
            "x1_norm": float(x1) / w,
            "y1_norm": float(y1) / h,
            "x2_norm": float(x2) / w,
            "y2_norm": float(y2) / h
        },
        "grid_x1": int(grid_x1),
        "grid_y1": int(grid_y1),
        "grid_x2": int(grid_x2),
        "grid_y2": int(grid_y2),
        "active_waveform_y1": int(final_y1),
        "active_waveform_y2": int(final_y2),
        "active_x1": int(final_x1),
        "active_x2": int(final_x2),
        "detected_grid_spacing_x": float(grid_spacing_x),
        "detected_grid_spacing_y": float(grid_spacing_y),
        "detected_row_count": len(genuine_rows),
        "detected_rows": detected_rows_out,
        "calibration_candidates": calibration_bboxes,
        "rejected_calibration_components": calibration_bboxes,
        "structural_grid_bbox": {
            "x1": int(grid_x1),
            "y1": int(grid_y1),
            "x2": int(grid_x2),
            "y2": int(grid_y2)
        },
        "active_ecg_panel_bbox": {
            "x1": int(final_x1),
            "y1": int(final_y1),
            "x2": int(final_x2),
            "y2": int(final_y2)
        },
        "delta": {
            "x1": int(x1 - initial_x1),
            "y1": int(y1 - initial_y1),
            "x2": int(x2 - initial_x2),
            "y2": int(y2 - initial_y2)
        },
        "boundary_status": {
            "left": status(initial_x1, x1),
            "top": status(initial_y1, y1),
            "right": status(initial_x2, x2),
            "bottom": status(initial_y2, y2)
        },
        "debug_images": debug_images
    }
