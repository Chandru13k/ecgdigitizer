import cv2
import numpy as np

def detect_ecg_panel(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Enhance grid lines using gradients
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # Threshold gradient
    _, binary_grad = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
    
    # Extract horizontal and vertical lines
    k_horiz = max(15, int(w * 0.01))
    k_vert = max(15, int(h * 0.01))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_horiz, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
    
    lines_h = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_v)
    grid_structure = cv2.bitwise_or(lines_h, lines_v)
    
    # 2. Build 1D projections of the grid evidence
    h_proj = np.sum(grid_structure, axis=1) / 255.0  # length h
    v_proj = np.sum(grid_structure, axis=0) / 255.0  # length w
    
    # Smooth projections heavily to create a continuous density profile
    smooth_y = int(h * 0.03) | 1
    smooth_x = int(w * 0.03) | 1
    
    h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(1, -1), (smooth_y, 1), 0).flatten()
    v_proj_smooth = cv2.GaussianBlur(v_proj.reshape(1, -1), (smooth_x, 1), 0).flatten()
    
    # Filter margins to avoid scanner artifacts
    margin_y = int(h * 0.02)
    margin_x = int(w * 0.02)
    h_proj_smooth[:margin_y] = 0
    h_proj_smooth[-margin_y:] = 0
    v_proj_smooth[:margin_x] = 0
    v_proj_smooth[-margin_x:] = 0
    
    # Find the peak (strongest interior grid region)
    peak_y = np.argmax(h_proj_smooth)
    peak_x = np.argmax(v_proj_smooth)
    
    # Search outward to find the continuous grid extent
    # Stop when the density drops below 5% of the peak (indicating the end of the grid)
    thresh_y = h_proj_smooth[peak_y] * 0.05
    thresh_x = v_proj_smooth[peak_x] * 0.05
    
    y1, y2 = peak_y, peak_y
    while y1 > 0 and h_proj_smooth[y1] > thresh_y:
        y1 -= 1
    while y2 < h - 1 and h_proj_smooth[y2] > thresh_y:
        y2 += 1
        
    x1, x2 = peak_x, peak_x
    while x1 > 0 and v_proj_smooth[x1] > thresh_x:
        x1 -= 1
    while x2 < w - 1 and v_proj_smooth[x2] > thresh_x:
        x2 += 1
        
    initial_x1, initial_y1, initial_x2, initial_y2 = x1, y1, x2, y2
    
    # 3. Waveform Safety Validation (Connected Component Expansion)
    # Ensure no ink (waveform) is clipped by the boundaries.
    # Dark ink mask
    _, ink_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    
    expanded = True
    max_expand_x = int(w * 0.03)
    max_expand_y = int(h * 0.03)
    
    while expanded:
        expanded = False
        # Get labels on the current boundaries
        top_edge = labels[max(0, y1):max(1, y1+1), max(0, x1):min(w, x2)]
        bottom_edge = labels[min(h-1, y2-1):min(h, y2), max(0, x1):min(w, x2)]
        left_edge = labels[max(0, y1):min(h, y2), max(0, x1):max(1, x1+1)]
        right_edge = labels[max(0, y1):min(h, y2), min(w-1, x2-1):min(w, x2)]
        
        border_labels = np.unique(np.concatenate([top_edge.flatten(), bottom_edge.flatten(), left_edge.flatten(), right_edge.flatten()]))
        
        for label in border_labels:
            if label == 0:
                continue # background
            
            # Get bounding box of this ink component
            comp_x = stats[label, cv2.CC_STAT_LEFT]
            comp_y = stats[label, cv2.CC_STAT_TOP]
            comp_w = stats[label, cv2.CC_STAT_WIDTH]
            comp_h = stats[label, cv2.CC_STAT_HEIGHT]
            comp_x2 = comp_x + comp_w
            comp_y2 = comp_y + comp_h
            
            # If the component expands our bounding box AND is within max bounds, update it
            if comp_x < x1 and comp_x >= initial_x1 - max_expand_x:
                x1 = comp_x
                expanded = True
            if comp_y < y1 and comp_y >= initial_y1 - max_expand_y:
                y1 = comp_y
                expanded = True
            if comp_x2 > x2 and comp_x2 <= initial_x2 + max_expand_x:
                x2 = comp_x2
                expanded = True
            if comp_y2 > y2 and comp_y2 <= initial_y2 + max_expand_y:
                y2 = comp_y2
                expanded = True
                
    # 4. Final safety margin (e.g., +0.5% padding) to avoid edge hugging
    pad_x = int(w * 0.005)
    pad_y = int(h * 0.005)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    # 5. Metrics Calculation
    panel_w = x2 - x1
    panel_h = y2 - y1
    if panel_w <= 0 or panel_h <= 0:
        return _fail_result(h, w)
        
    panel_area_ratio = (panel_w * panel_h) / (w * h)
    
    roi_grad = binary_grad[y1:y2, x1:x2]
    grid_density = np.sum(roi_grad) / 255.0 / (panel_w * panel_h)
    
    roi_ink = ink_mask[y1:y2, x1:x2]
    waveform_density = np.sum(roi_ink) / 255.0 / (panel_w * panel_h)
    
    waveform_clipping_detected = (x1 < initial_x1 or y1 < initial_y1 or x2 > initial_x2 or y2 > initial_y2)
    
    overall_score = (grid_density * 2) + (panel_area_ratio * 0.5)
    
    status = "PASS"
    panel_completeness_check = "PASS"
    if panel_area_ratio < 0.2:
        status = "WARN"
    if panel_area_ratio < 0.05:
        status = "FAIL"
        panel_completeness_check = "FAIL"
        
    unrelated_content_ratio = 1.0 - panel_area_ratio
    
    import hashlib
    # Compute SHA256 of the source image to track pipeline integrity
    sha256_hash = hashlib.sha256(image.tobytes()).hexdigest()
    
    return {
        "stage": "stage2_1",
        "detector_version": "stage2.1",
        "source_sha256": sha256_hash,
        "image_width": w,
        "image_height": h,
        "config": {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "max_expand_x": max_expand_x,
            "max_expand_y": max_expand_y
        },
        "ecg_panel_bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        },
        "ecg_panel_width": int(panel_w),
        "ecg_panel_height": int(panel_h),
        "panel_area_ratio": float(panel_area_ratio),
        "grid_density": float(grid_density),
        "waveform_density": float(waveform_density),
        "waveform_clipping_detected": bool(waveform_clipping_detected),
        "panel_completeness_check": panel_completeness_check,
        "overall_detection_score": float(overall_score),
        "boundary_margin": {
            "left": int(initial_x1 - x1),
            "right": int(x2 - initial_x2),
            "top": int(initial_y1 - y1),
            "bottom": int(y2 - initial_y2)
        },
        "estimated_unrelated_content_ratio": float(unrelated_content_ratio),
        "status": status,
        "debug_images": {
            "grid_evidence": grid_structure,
            "binary_grad": binary_grad
        }
    }

def _fail_result(h, w):
    return {
        "stage": "stage2_1",
        "detector_version": "stage2.1",
        "source_sha256": "",
        "image_width": w,
        "image_height": h,
        "config": {},
        "ecg_panel_bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
        "ecg_panel_width": w,
        "ecg_panel_height": h,
        "panel_area_ratio": 1.0,
        "grid_density": 0.0,
        "waveform_density": 0.0,
        "waveform_clipping_detected": False,
        "panel_completeness_check": "FAIL",
        "overall_detection_score": 0.0,
        "boundary_margin": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "estimated_unrelated_content_ratio": 0.0,
        "status": "FAIL",
        "debug_images": {}
    }
