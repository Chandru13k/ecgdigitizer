import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

def extract_waveform_coordinates(binary_patch, config):
    h, w = binary_patch.shape
    
    signal = np.full(w, np.nan)
    
    # 1. Column-wise analysis (Strict NaN handling)
    for x in range(w):
        ys = np.where(binary_patch[:, x] > 0)[0]
        if len(ys) > 0:
            # Simple median for now. In run_stage5_2 we do subpixel, this is a fallback
            c = np.median(ys)
            signal[x] = c
            
    # 2. DO NOT INTERPOLATE
    # Leave gaps as NaN
    
    # 3. DO NOT SMOOTH
    
    valid_sig = ~np.isnan(signal)
    
    # Invert y-axis so that higher values mean higher physiological amplitude
    # We only invert valid values
    signal[valid_sig] = h - signal[valid_sig]
    
    metrics = {
        "continuity": float(np.sum(valid_sig) / w),
        "interpolated_ratio": 0.0, # Strictly zero interpolation
        "straight_line_ratio": 0.0,
        "grid_contamination": 0.0
    }
    
    vis = (binary_patch.copy()).astype(np.uint8)
    if vis.max() == 1:
        vis = vis * 255
        
    return signal, metrics, vis
