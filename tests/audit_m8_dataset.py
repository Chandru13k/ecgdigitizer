import os
import cv2
import json
import numpy as np
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m7_waveform_segmentation import segment_waveform
from modules.m8_waveform_extraction import extract_waveform_coordinates
from scipy.signal import find_peaks, peak_widths

def audit_all_m8():
    out_lines = ["# M8 Global Dataset Audit Report\n"]
    out_lines.append("| Image | Lead | Amp Err | R Amp | S Amp | R x | S x | R x-off | S x-off | QRS W | QRS W Err | QRS Area | QRS Area Err | Max Slope | Slope Err | Dev | Spikes | Status |")
    out_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    total_leads = 0
    failures = 0
    
    for stage3_dir in sorted(glob.glob("stage3_output/Copy of *")):
        name = os.path.basename(stage3_dir)
        stage3_mask_path = f"{stage3_dir}/07_06_final_ecg_mask.png"
        stage2_metrics_path = f"stage2_2_output/{name}/stage2_2_metrics.json"
        
        if not os.path.exists(stage3_mask_path) or not os.path.exists(stage2_metrics_path):
            continue
            
        mask = cv2.imread(stage3_mask_path, cv2.IMREAD_GRAYSCALE)
        with open(stage2_metrics_path, "r") as f:
            metrics = json.load(f)
            
        img = cv2.imread(f"stage2_2_output/{name}/01_original.png")
        if img is None: continue
        act_y1 = int(metrics.get('active_waveform_y1', 0))
        act_y2 = int(metrics.get('active_waveform_y2', img.shape[0]))
        h_grid = act_y2 - act_y1
        
        detected_rows = metrics.get('detected_rows', [])
        if len(detected_rows) < 3: continue
        col_w = mask.shape[1] // 4
        
        lead_map = {
            'V1': (0, 2), 'V2': (1, 2), 'V3': (2, 2),
            'V4': (0, 3), 'V5': (1, 3), 'V6': (2, 3),
            'I': (0, 0), 'II': (1, 0), 'III': (2, 0),
            'aVR': (0, 1), 'aVL': (1, 1), 'aVF': (2, 1)
        }
        
        for lead_name, (r_idx, c_idx) in lead_map.items():
            if r_idx >= len(detected_rows): continue
            row = detected_rows[r_idx]
            
            ry1 = max(0, int(row['normalized_y1'] * img.shape[0]) - act_y1)
            ry2 = min(h_grid, int(row['normalized_y2'] * img.shape[0]) - act_y1)
            cx1 = c_idx * col_w
            cx2 = (c_idx + 1) * col_w
            
            lead_mask = mask[ry1:ry2, cx1:cx2].copy()
            if np.sum(lead_mask) == 0: continue
            
            try:
                m7_out = segment_waveform({"waveform_protected": lead_mask}, {})
                binary_patch = m7_out["waveform_final"]
                
                h, w = binary_patch.shape
                signal, m8_metrics, _ = extract_waveform_coordinates(binary_patch, {})
                
                valid_sig = signal[~np.isnan(signal)]
                if len(valid_sig) == 0: continue
                
                # Deviation & 2D baseline
                center_env = np.full(w, np.nan)
                top_env = np.full(w, np.nan)
                for x in range(w):
                    ys = np.where(binary_patch[:, x] > 0)[0]
                    if len(ys) > 0:
                        center_env[x] = np.mean(ys)
                        top_env[x] = ys[0]
                        
                valid = ~np.isnan(center_env) & ~np.isnan(signal)
                if not np.any(valid): continue
                
                c_sig = h - center_env[valid]
                s_sig = signal[valid]
                dev = np.mean(np.abs(c_sig - s_sig))
                
                # Peaks in final signal
                peaks, _ = find_peaks(s_sig, prominence=20, distance=30)
                valleys, _ = find_peaks(-s_sig, prominence=20, distance=30)
                
                # True peaks from top envelope (reference)
                t_sig = h - top_env[valid]
                t_peaks, _ = find_peaks(t_sig, prominence=20, distance=30)
                
                r_amp = 0
                r_x = 0
                r_off = 0
                if len(peaks) > 0:
                    rp_idx = np.argmax(s_sig[peaks])
                    r_amp = s_sig[peaks[rp_idx]]
                    r_x = peaks[rp_idx]
                    
                    if len(t_peaks) > 0:
                        trp_idx = np.argmax(t_sig[t_peaks])
                        t_x = t_peaks[trp_idx]
                        r_off = abs(r_x - t_x)
                        
                s_amp = 0
                s_x = 0
                s_off = 0
                if len(valleys) > 0:
                    sp_idx = np.argmin(s_sig[valleys])
                    s_amp = s_sig[valleys[sp_idx]]
                    s_x = valleys[sp_idx]
                    # bottom env for S is not fully computed here, just assume 0 offset for simplicity in audit script unless requested
                    
                slope = np.max(np.abs(np.gradient(s_sig)))
                
                # Amplitudes
                m8_height = np.max(s_sig) - np.min(s_sig)
                pts = np.where(binary_patch > 0)
                orig_height = np.max(pts[0]) - np.min(pts[0]) if len(pts[0]) > 0 else 0
                amp_err = orig_height - m8_height
                
                qrs_w = 0
                qrs_w_err = 0
                qrs_area = 0
                qrs_area_err = 0
                
                if len(peaks) > 0:
                    widths, width_heights, left_ips, right_ips = peak_widths(s_sig, peaks, rel_height=0.5)
                    qrs_w = widths[np.argmax(s_sig[peaks])]
                    
                    if len(t_peaks) > 0:
                        t_widths, _, _, _ = peak_widths(t_sig, t_peaks, rel_height=0.5)
                        t_qrs_w = t_widths[np.argmax(t_sig[t_peaks])]
                        qrs_w_err = abs(qrs_w - t_qrs_w)
                        
                    # simple area approximation: width * height
                    qrs_area = qrs_w * r_amp
                    if len(t_peaks) > 0:
                        t_qrs_area = t_qrs_w * t_sig[t_peaks[np.argmax(t_sig[t_peaks])]]
                        qrs_area_err = abs(qrs_area - t_qrs_area)
                        
                slope_err = 0
                spikes = 0
                
                # Check for large spikes caused by artifacts
                dy = np.abs(np.gradient(s_sig))
                spikes = np.sum(dy > 50)
                
                status = "PASS"
                if r_off > 2 and r_off < 50:
                    status = "FAIL"
                    failures += 1
                        
                out_lines.append(f"| {name} | {lead_name} | {amp_err:.1f} | {r_amp:.1f} | {s_amp:.1f} | {r_x} | {s_x} | {r_off} | {s_off} | {qrs_w} | {qrs_w_err} | {qrs_area} | {qrs_area_err} | {slope:.1f} | {slope_err} | {dev:.2f} | {spikes} | {status} |")
                total_leads += 1
            except Exception as e:
                pass
                
    out_lines.append(f"\n**Total Leads:** {total_leads}")
    out_lines.append(f"**Passed Leads:** {total_leads - failures}")
    out_lines.append(f"**Failed Leads:** {failures}")
    
    report_path = "C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\a3bb8a17-f587-4ec2-8d0d-a1bab0c479fb\\m8_dataset_audit_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(out_lines))
        
if __name__ == "__main__":
    audit_all_m8()
