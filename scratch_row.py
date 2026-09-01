import cv2
import numpy as np
import json
import sys

def test_level_b(image_path, json_path):
    print(f"\n--- Testing {image_path} ---")
    image = cv2.imread(str(image_path))
    with open(json_path, 'r') as f:
        metrics = json.load(f)
        
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Spacing
    initial_k_horiz = max(5, int(w * 0.005))
    initial_k_vert = max(5, int(h * 0.005))
    kernel_h_init = cv2.getStructuringElement(cv2.MORPH_RECT, (initial_k_horiz, 1))
    kernel_v_init = cv2.getStructuringElement(cv2.MORPH_RECT, (1, initial_k_vert))
    
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    _, binary_grad = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
    
    lines_h_init = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_h_init)
    lines_v_init = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_v_init)
    initial_grid = cv2.bitwise_or(lines_h_init, lines_v_init)
    
    raw_h_proj = np.sum(initial_grid, axis=1) / 255.0
    raw_v_proj = np.sum(initial_grid, axis=0) / 255.0
    
    def estimate_spacing(proj):
        proj_smooth = np.convolve(proj, np.ones(5)/5.0, mode='same')
        peaks = []
        for i in range(1, len(proj_smooth)-1):
            if proj_smooth[i] > proj_smooth[i-1] and proj_smooth[i] > proj_smooth[i+1] and proj_smooth[i] > np.max(proj_smooth)*0.1:
                peaks.append(i)
        merged = []
        for p in peaks:
            if not merged or p - merged[-1] > 10:
                merged.append(p)
        if len(merged) > 1: return np.median(np.diff(merged))
        return max(5.0, min(h, w) * 0.01)
        
    gs_y = estimate_spacing(raw_h_proj)
    gs_x = estimate_spacing(raw_v_proj)
    print(f"Spacing: x={gs_x}, y={gs_y}")
    
    # Grid
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    _, binary_grad = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)
    
    k_horiz = max(3, int(gs_x * 2.5))
    k_vert = max(3, int(gs_y * 2.5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_horiz, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_vert))
    lines_h = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_h)
    lines_v = cv2.morphologyEx(binary_grad, cv2.MORPH_OPEN, kernel_v)
    grid_structure = cv2.bitwise_or(lines_h, lines_v)
    
    # LEVEL B - INK (Strict threshold to ignore pink grid and only keep dark black ink)
    _, pure_ink_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # subtract grid lines
    # ink_no_grid = cv2.bitwise_and(ink_mask, cv2.bitwise_not(grid_structure))
    
    # Restrict to structural bounds
    bbox = metrics["ecg_panel_bbox"]
    gx1, gy1, gx2, gy2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    roi_ink = np.zeros_like(pure_ink_mask)
    roi_ink[gy1:gy2, gx1:gx2] = pure_ink_mask[gy1:gy2, gx1:gx2]
    
    # find y_min and y_max for each row using original pure ink mask connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pure_ink_mask, connectivity=8)
    
    valid_bboxes = []
    
    for lbl in range(1, num_labels):
        cx = stats[lbl, cv2.CC_STAT_LEFT]
        cy = stats[lbl, cv2.CC_STAT_TOP]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        carea = stats[lbl, cv2.CC_STAT_AREA]
        
        # Must be inside the structural grid
        if cy < gy1 or cy + ch > gy2 or cx < gx1 or cx + cw > gx2:
            continue
            
        fill_ratio = carea / (cw * ch + 1e-5)
        
        is_trace = cw > gs_x * 3.0 and fill_ratio < 0.4
        is_qrs = ch > gs_y * 2.0 and cw < gs_x * 2.5 and fill_ratio < 0.35
        is_complex = cw > gs_x * 2.5 and ch > gs_y * 1.5 and fill_ratio < 0.35
        
        if ch > gs_y * 4 or cw > gs_x * 4:
            print(f"Component at cy={cy}, cx={cx}, cw={cw}, ch={ch}, carea={carea}, fill={fill_ratio}")
            
        if is_trace or is_qrs or is_complex:
            valid_bboxes.append((cx, cy, cw, ch))
            
    # Cluster valid bboxes by y-center to find rows
    row_centers = []
    for cx, cy, cw, ch in valid_bboxes:
        mid_y = cy + ch // 2
        # check if it belongs to an existing row
        found = False
        for i, (rc, bboxes) in enumerate(row_centers):
            if abs(rc - mid_y) < gs_y * 3:
                # Update row center as weighted average or just keep track
                row_centers[i][1].append((cx, cy, cw, ch))
                found = True
                break
        if not found:
            row_centers.append((mid_y, [(cx, cy, cw, ch)]))
            
    # Sort rows by y
    row_centers.sort(key=lambda x: x[0])
    
    # Filter out spurious rows with very few components or narrow horizontal extent
    genuine_rows = []
    structural_width = gx2 - gx1
    for rc, bboxes in row_centers:
        min_x = min(b[0] for b in bboxes)
        max_x = max(b[0] + b[2] for b in bboxes)
        total_w = max_x - min_x
        
        # A primary ECG row spans a large portion of the structural grid
        if total_w > structural_width * 0.6: 
            genuine_rows.append((rc, bboxes))
            
    print(f"Detected {len(genuine_rows)} genuine rows")
    for i, (rc, bboxes) in enumerate(genuine_rows):
        min_y = min(b[1] for b in bboxes)
        max_y = max(b[1] + b[3] for b in bboxes)
        min_x = min(b[0] for b in bboxes)
        max_x = max(b[0] + b[2] for b in bboxes)
        print(f"Row {i} (center {rc:.1f}): y({min_y}-{max_y}), x({min_x}-{max_x})")

if __name__ == "__main__":
    test_level_b("stage1_output/Copy of 20_male/page_001.png", "stage2_1_output/Copy of 20_male/stage2_1_metrics.json")
    test_level_b("stage1_output/Copy of 19_male/page_001.png", "stage2_1_output/Copy of 19_male/stage2_1_metrics.json")
