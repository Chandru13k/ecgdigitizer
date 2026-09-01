import cv2
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m7_waveform_segmentation import segment_waveform
from modules.m8_waveform_extraction import extract_waveform_coordinates

def calculate_metrics(signal, center_env, top_env, bot_env, valid_idx, h):
    # This is physical amplitude (h - y)
    valid_sig = signal[valid_idx]
    
    # 1. Peaks (R, T)
    peaks, _ = find_peaks(valid_sig, prominence=20, distance=30)
    # Valleys (S)
    valleys, _ = find_peaks(-valid_sig, prominence=20, distance=30)
    
    # Sort peaks by amplitude
    peaks_sorted = sorted(peaks, key=lambda i: valid_sig[i], reverse=True)
    valleys_sorted = sorted(valleys, key=lambda i: valid_sig[i])
    
    r_peak = peaks_sorted[0] if peaks_sorted else None
    s_valley = valleys_sorted[0] if valleys_sorted else None
    
    # T wave is usually the second highest peak or next prominent
    t_peak = None
    if len(peaks_sorted) > 1:
        # Just pick the next prominent for now or farthest one
        # This is a simplification for audit
        t_peak = peaks_sorted[1]
        
    metrics = {}
    
    if r_peak is not None:
        metrics['r_peak_amp'] = valid_sig[r_peak]
        metrics['r_peak_x'] = valid_idx[r_peak]
        
        # Center peak phase shift
        center_sig = h - center_env[valid_idx]
        c_peaks, _ = find_peaks(center_sig, prominence=20, distance=30)
        if len(c_peaks) > 0:
            c_r = sorted(c_peaks, key=lambda i: center_sig[i], reverse=True)[0]
            metrics['r_peak_phase_shift'] = abs(valid_idx[r_peak] - valid_idx[c_r])
            
    if s_valley is not None:
        metrics['s_valley_amp'] = valid_sig[s_valley]
        metrics['s_valley_x'] = valid_idx[s_valley]
        
    if t_peak is not None:
        metrics['t_peak_amp'] = valid_sig[t_peak]
        metrics['t_peak_x'] = valid_idx[t_peak]
        
    # Max slope
    slope = np.gradient(valid_sig)
    metrics['max_slope'] = np.max(np.abs(slope))
    
    # Centerline deviation (MSE)
    center_sig = h - center_env[valid_idx]
    metrics['centerline_deviation'] = np.mean(np.abs(valid_sig - center_sig))
    
    return metrics, peaks, valleys

def audit_morphology(name, lead_name="V2"):
    print(f"Auditing {name} Lead {lead_name}")
    
    stage3_mask_path = f"stage3_output/{name}/07_06_final_ecg_mask.png"
    if not os.path.exists(stage3_mask_path):
        print(f"Missing mask for {name}")
        return
        
    stage2_metrics_path = f"stage2_2_output/{name}/stage2_2_metrics.json"
    if not os.path.exists(stage2_metrics_path):
        print(f"Missing metrics for {name}")
        return
        
    with open(stage2_metrics_path, "r") as f:
        metrics = json.load(f)
        
    mask = cv2.imread(stage3_mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(f"stage2_2_output/{name}/01_original.png")
    
    act_y1 = int(metrics.get('active_waveform_y1', 0))
    act_y2 = int(metrics.get('active_waveform_y2', img.shape[0]))
    h_grid = act_y2 - act_y1
    
    detected_rows = metrics.get('detected_rows', [])
    col_w = mask.shape[1] // 4
    
    lead_map = {
        'V1': (0, 2), 'V2': (1, 2), 'V3': (2, 2),
        'V4': (0, 3), 'V5': (1, 3), 'V6': (2, 3),
        'I': (0, 0), 'II': (1, 0), 'III': (2, 0),
        'aVR': (0, 1), 'aVL': (1, 1), 'aVF': (2, 1)
    }
    
    if lead_name not in lead_map:
        return
        
    r_idx, c_idx = lead_map[lead_name]
    if r_idx >= len(detected_rows): return
    row = detected_rows[r_idx]
    
    ry1 = max(0, int(row['normalized_y1'] * img.shape[0]) - act_y1)
    ry2 = min(h_grid, int(row['normalized_y2'] * img.shape[0]) - act_y1)
    cx1 = c_idx * col_w
    cx2 = (c_idx + 1) * col_w
    
    lead_mask = mask[ry1:ry2, cx1:cx2].copy()
    
    # Run M7
    m7_out = segment_waveform({"waveform_protected": lead_mask}, {})
    binary_patch = m7_out["waveform_final"]
    
    # Run M8
    h, w = binary_patch.shape
    signal, m8_metrics, _ = extract_waveform_coordinates(binary_patch, {})
    
    # Envelopes
    top_env = np.full(w, np.nan)
    bot_env = np.full(w, np.nan)
    center_env = np.full(w, np.nan)
    for x in range(w):
        ys = np.where(binary_patch[:, x] > 0)[0]
        if len(ys) > 0:
            top_env[x] = ys[0]
            bot_env[x] = ys[-1]
            center_env[x] = np.mean(ys)
            
    valid = ~np.isnan(signal)
    if not np.any(valid):
        print("No valid signal")
        return
        
    xs = np.arange(w)[valid]
    valid_sig = signal[valid]
    
    calc_metrics, peaks, valleys = calculate_metrics(signal, center_env, top_env, bot_env, xs, h)
    
    print("--- Metrics ---")
    for k, v in calc_metrics.items():
        print(f"{k}: {v:.2f}")
        
    # Plotting
    plt.figure(figsize=(15, 6), dpi=300)
    
    img_plot = np.zeros((h, w, 3), dtype=np.uint8)
    img_plot[binary_patch > 0] = [200, 200, 200]
    
    plt.imshow(img_plot, extent=[0, w, 0, h], origin='lower', aspect='auto')
    
    plt.plot(xs, valid_sig, color='green', linewidth=1.5, label='M8 1D Signal')
    
    valid_c = ~np.isnan(center_env)
    if np.any(valid_c):
        plt.plot(np.arange(w)[valid_c], h - center_env[valid_c], 'y--', linewidth=1, label='Center of Mass')
        
    valid_t = ~np.isnan(top_env)
    if np.any(valid_t):
        plt.plot(np.arange(w)[valid_t], h - top_env[valid_t], 'r--', linewidth=1, label='Top Envelope')
        
    valid_b = ~np.isnan(bot_env)
    if np.any(valid_b):
        plt.plot(np.arange(w)[valid_b], h - bot_env[valid_b], 'b--', linewidth=1, label='Bottom Envelope')
        
    # Mark peaks
    if len(peaks) > 0:
        plt.plot(xs[peaks], valid_sig[peaks], 'r^', markersize=10, label='Peaks')
    if len(valleys) > 0:
        plt.plot(xs[valleys], valid_sig[valleys], 'bv', markersize=10, label='Valleys')
        
    plt.title(f"{name} {lead_name} Morphology Overlay")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('tests/morphology_audits', exist_ok=True)
    out_path = f"tests/morphology_audits/{name}_{lead_name}_audit.png"
    plt.savefig(out_path)
    print(f"Saved diagnostic overlay to {out_path}\n")
    plt.close()

if __name__ == "__main__":
    for p in ["20_male", "19_male"]:
        for l in ["V2", "V3", "aVR", "aVF"]:
            # Only run ones that exist (20_male V2/V3/aVR/aVF, 19_male V2/V3)
            if p == "19_male" and l in ["aVR", "aVF"]: continue
            audit_morphology(f"Copy of {p}", l)
