import cv2
import numpy as np

def segment_waveform(waveform_dict, config):
    waveform_protected = waveform_dict["waveform_protected"]
    h, w = waveform_protected.shape[:2]
    
    # 1. Adaptive Gap Bridging
    # We only bridge very small gaps consistent with grid intersections.
    gap_k = max(2, int(min(h, w) * 0.008)) 
    gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_k, gap_k))
    
    waveform_bridged = cv2.morphologyEx(waveform_protected, cv2.MORPH_CLOSE, gap_kernel)
    
    # 2. Filter out isolated noise/dots (but keep it small so we don't erase fragmented valid trace)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(waveform_bridged, connectivity=8)
    
    # Adaptive min size based on area
    min_size = max(10, int((h * w) * 0.0001))
    
    waveform_final = np.zeros_like(waveform_bridged)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            waveform_final[labels == i] = 255
            
    # Copy all masks for debug/diagnostics
    waveform_dict["waveform_final"] = waveform_final
    return waveform_dict
