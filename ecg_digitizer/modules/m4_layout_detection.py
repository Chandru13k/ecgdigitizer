import numpy as np
import cv2

def detect_ecg_layout(color_img, config):
    h, w, _ = color_img.shape
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Simple horizontal/vertical projection to find grid bounds
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    
    # Horizontal projection
    h_proj = np.sum(thresh, axis=1)
    # Vertical projection
    v_proj = np.sum(thresh, axis=0)
    
    # Find start and end of grid
    h_idx = np.where(h_proj > w * 255 * 0.05)[0]
    v_idx = np.where(v_proj > h * 255 * 0.05)[0]
    
    if len(h_idx) > 0 and len(v_idx) > 0:
        pad_top = max(0, h_idx[0] - 20)
        pad_bottom = h - min(h, h_idx[-1] + 20)
        pad_left = max(0, v_idx[0] - 20)
        pad_right = w - min(w, v_idx[-1] + 20)
    else:
        pad_top = int(h * 0.20)
        pad_bottom = int(h * 0.05)
        pad_left = int(w * 0.05)
        pad_right = int(w * 0.02)
        
    ch = h - pad_top - pad_bottom
    cw = w - pad_left - pad_right
    
    row_h = ch // 4
    col_w = cw // 4
    
    lead_names = config['lead_names']
    bboxes = {}
    
    for r in range(3):
        for c in range(4):
            name = lead_names[r][c]
            ymin = pad_top + r * row_h
            ymax = pad_top + (r + 1) * row_h
            xmin = pad_left + c * col_w
            xmax = pad_left + (c + 1) * col_w
            bboxes[name] = (ymin, xmin, ymax, xmax)
            
    rhythm_name = config['rhythm_lead']
    ymin_r = pad_top + 3 * row_h
    ymax_r = pad_top + 4 * row_h
    xmin_r = pad_left
    xmax_r = pad_left + cw
    bboxes[rhythm_name] = (ymin_r, xmin_r, ymax_r, xmax_r)
    
    vis = color_img.copy()
    for name, (ymin, xmin, ymax, xmax) in bboxes.items():
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(vis, name, (xmin + 10, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
    return bboxes, vis
