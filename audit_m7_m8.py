import cv2
import numpy as np
import json
import os
import sys

# Add modules to path so we can import them
sys.path.append(os.path.join(os.path.dirname(__file__), 'ecg_digitizer'))
from modules.m7_waveform_segmentation import segment_waveform
from modules.m8_waveform_extraction import extract_waveform_coordinates

def audit_m7_m8(name):
    print(f"Auditing m7 and m8 for {name}...")
    
    # Load Stage 3.1 output
    out_dir = f"stage3_output/{name}"
    final_mask = cv2.imread(f"{out_dir}/07_06_final_ecg_mask.png", cv2.IMREAD_GRAYSCALE)
    if final_mask is None:
        print("Final mask not found!")
        return
        
    with open(f"stage2_2_output/{name}/stage2_2_metrics.json", "r") as f:
        metrics = json.load(f)
        
    # Isolate V2 for analysis
    img = cv2.imread(f"stage2_2_output/{name}/01_original.png")
    act_y1 = int(metrics.get('active_waveform_y1', 0))
    act_y2 = int(metrics.get('active_waveform_y2', img.shape[0]))
    h = act_y2 - act_y1
    
    detected_rows = metrics.get('detected_rows', [])
    col_w = final_mask.shape[1] // 4
    
    # V2 is at row 2 (index 1), col 3 (index 2)
    row = detected_rows[1]
    ry1 = max(0, int(row['normalized_y1'] * img.shape[0]) - act_y1)
    ry2 = min(h, int(row['normalized_y2'] * img.shape[0]) - act_y1)
    cx1 = 2 * col_w
    cx2 = 3 * col_w
    
    v2_mask = final_mask[ry1:ry2, cx1:cx2].copy()
    
    # Original V2 height
    pts = np.where(v2_mask > 0)
    orig_min_y = np.min(pts[0]) if len(pts[0]) > 0 else 0
    orig_max_y = np.max(pts[0]) if len(pts[0]) > 0 else 0
    orig_height = orig_max_y - orig_min_y
    print(f"[Stage 3.1] V2 Original Height: {orig_height}")
    
    cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_m7_input.png", v2_mask)
    
    # --- M7 AUDIT ---
    # Apply m7 operations manually to observe each step
    
    # 1. Morphological close
    gap_k = max(2, int(min(ry2-ry1, cx2-cx1) * 0.008))
    gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_k, gap_k))
    v2_bridged = cv2.morphologyEx(v2_mask, cv2.MORPH_CLOSE, gap_kernel)
    
    cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_m7_after_morphology.png", v2_bridged)
    
    # Check pixels added/removed by closing
    added_by_close = np.sum((v2_bridged > 0) & (v2_mask == 0))
    print(f"[m7] Pixels added by MORPH_CLOSE: {added_by_close}")
    
    # 2. CC filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(v2_bridged, connectivity=8)
    min_size = max(10, int(((ry2-ry1)*(cx2-cx1)) * 0.0001))
    
    v2_final = np.zeros_like(v2_bridged)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            v2_final[labels == i] = 255
            
    cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_m7_final_mask.png", v2_final)
    
    # Check pixels removed by CC filter
    removed_by_cc = np.sum((v2_bridged > 0) & (v2_final == 0))
    print(f"[m7] Pixels removed by CC Area Filter (<{min_size}): {removed_by_cc}")
    
    # M7 Output height
    pts_m7 = np.where(v2_final > 0)
    m7_min_y = np.min(pts_m7[0]) if len(pts_m7[0]) > 0 else 0
    m7_max_y = np.max(pts_m7[0]) if len(pts_m7[0]) > 0 else 0
    m7_height = m7_max_y - m7_min_y
    print(f"[m7] V2 Height after m7: {m7_height}")
    
    # --- M8 AUDIT ---
    # Apply m8
    config = {'dp_weight_jump': 1.0}
    signal, m8_metrics, skeleton = extract_waveform_coordinates(v2_final, config)
    
    cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_m8_skeleton.png", skeleton)
    
    # M8 Extracted height
    # The signal array contains the y coordinates, which are inverted in m8 (signal = h - signal).
    # Wait, the shape of the patch is h, w. So h - signal is the y coordinate from top.
    # The minimum and maximum valid y coordinates
    valid_sig = signal[~np.isnan(signal)]
    m8_min_y = np.min(valid_sig) if len(valid_sig) > 0 else 0
    m8_max_y = np.max(valid_sig) if len(valid_sig) > 0 else 0
    m8_height = m8_max_y - m8_min_y
    print(f"[m8] V2 Height extracted by m8 DP algorithm: {m8_height}")
    print(f"[m8] m8_min_y = {m8_min_y}, m8_max_y = {m8_max_y}")
    print(f"[m8] orig_min_y = {orig_min_y}, orig_max_y = {orig_max_y}")
    print(f"[m8] Amplitude Loss: {orig_height - m8_height} pixels ({(orig_height - m8_height)/orig_height*100:.1f}%)")
    
    # Reconstruct extracted signal for visualization
    m8_reconstruct = np.zeros_like(v2_final)
    for x in range(len(signal)):
        if not np.isnan(signal[x]):
            y = int(signal[x])
            if 0 <= y < m8_reconstruct.shape[0]:
                cv2.circle(m8_reconstruct, (x, y), 2, 255, -1)
                
    cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_m8_extracted_signal.png", m8_reconstruct)

if __name__ == "__main__":
    audit_m7_m8("Copy of 20_male")
