import sys
import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    HAS_NK = False

def extract_branch_a(img_roi, mask_roi):
    """
    Branch A: Immutable high-fidelity extraction using grayscale subpixel intensity weighting.
    """
    h, w = mask_roi.shape
    waveform = []
    
    for x in range(w):
        col_mask = mask_roi[:, x]
        y_coords = np.where(col_mask > 0)[0]
        
        if len(y_coords) > 0:
            intensities = img_roi[y_coords, x].astype(np.float32)
            # Ensure background (white) has low weight, waveform (black) has high weight
            weights = 255.0 - intensities
            # To prevent divide by zero if completely white (which shouldn't happen in mask)
            weights = np.maximum(weights, 1.0) 
            
            y_subpixel = np.sum(y_coords * weights) / np.sum(weights)
            y_center = float(np.median(y_coords))
            width = len(y_coords)
            
            waveform.append({
                "x_pixel": x,
                "raw_y_subpixel": y_subpixel,
                "foreground_width": width,
                "confidence": 1.0,
                "valid": True
            })
        else:
            waveform.append({
                "x_pixel": x,
                "raw_y_subpixel": None,
                "foreground_width": 0,
                "confidence": 0.0,
                "valid": False
            })
            
    return pd.DataFrame(waveform)

def calculate_morphology_features(df_waveform):
    """
    Calculates point-level morphology on an analysis copy.
    Master trace remains untouched.
    """
    df = df_waveform.copy()
    
    # Analysis copy for features
    y_vals = df['raw_y_subpixel'].values.copy()
    valid = df['valid'].values
    
    # Simple linear interpolation just for the analysis copy to compute continuous derivatives
    xs = df['x_pixel'].values
    valid_idx = np.where(valid)[0]
    
    if len(valid_idx) > 5:
        y_interp = np.interp(xs, xs[valid_idx], y_vals[valid_idx])
        # Smooth copy for derivatives
        window = min(11, len(y_interp) - (len(y_interp) % 2 == 0))
        if window >= 3:
            y_smooth = medfilt(y_interp, window)
            dy = np.gradient(y_smooth)
            d2y = np.gradient(dy)
            
            # Curvature k = |y''| / (1 + y'^2)^(3/2)
            curvature = np.abs(d2y) / (1 + dy**2)**1.5
            
            df['dy_dx'] = dy
            df['d2y_dx2'] = d2y
            df['curvature'] = curvature
        else:
            df['dy_dx'] = np.nan
            df['d2y_dx2'] = np.nan
            df['curvature'] = np.nan
    else:
        df['dy_dx'] = np.nan
        df['d2y_dx2'] = np.nan
        df['curvature'] = np.nan
        
    return df

def detect_branch_b(df_waveform):
    """
    Branch B: Landmark Detection on derived signals.
    """
    # 1. Convert to signal
    valid = df_waveform.dropna(subset=['raw_y_subpixel']).copy()
    if len(valid) < 50:
        return pd.DataFrame(), pd.DataFrame() # Not enough data
        
    x_val = valid['x_pixel'].values
    # Invert for physiological polarity
    raw_signal = -valid['raw_y_subpixel'].values
    
    # Very short gaps (1-2px) can be interpolated for continuous deriv, 
    # but we will just operate on the valid points directly to keep it simple,
    # mapping indices back to x_val.
    
    # 2. Baseline estimation (Median filter ~100px window)
    window = min(101, len(raw_signal) - (len(raw_signal) % 2 == 0))
    if window < 3: window = 3
    baseline = medfilt(raw_signal, window)
    baseline_corrected = raw_signal - baseline
    
    # 3. QRS Enhancement (First derivative + energy)
    derivative = np.gradient(baseline_corrected)
    qrs_energy = derivative ** 2
    # Moving average integration (Pan-Tompkins style)
    integ_window = min(30, len(qrs_energy))
    qrs_energy_smooth = np.convolve(qrs_energy, np.ones(integ_window)/integ_window, mode='same')
    
    # 4. Adaptive Thresholding
    threshold = np.mean(qrs_energy_smooth) + 1.5 * np.std(qrs_energy_smooth)
    qrs_candidates, _ = find_peaks(qrs_energy_smooth, height=threshold, distance=50)
    
    landmarks = []
    
    # NeuroKit2 validation
    nk_r_peaks = []
    if HAS_NK and len(baseline_corrected) > 100:
        try:
            # We treat the baseline_corrected as a raw ECG signal sampled at an arbitrary rate (e.g. 250Hz)
            # just to get an independent peak detection.
            _, info = nk.ecg_peaks(baseline_corrected, sampling_rate=250)
            nk_r_peaks = info["ECG_R_Peaks"]
        except:
            pass

    for i, peak_idx in enumerate(qrs_candidates):
        # Search window for actual R/S on the baseline_corrected signal
        left = max(0, peak_idx - 30)
        right = min(len(baseline_corrected)-1, peak_idx + 30)
        window_sig = baseline_corrected[left:right]
        window_x = x_val[left:right]
        
        # Find local maximum (R peak) and local minimum (S nadir)
        if len(window_sig) > 0:
            local_max_idx = np.argmax(window_sig)
            local_min_idx = np.argmin(window_sig)
            
            r_x = window_x[local_max_idx]
            r_y = raw_signal[left + local_max_idx]
            
            s_x = window_x[local_min_idx]
            s_y = raw_signal[left + local_min_idx]
            
            max_amp = window_sig[local_max_idx]
            min_amp = window_sig[local_min_idx]
            
            # Consensus logic for R peak
            nk_agree = any(abs((left + local_max_idx) - nk_p) < 10 for nk_p in nk_r_peaks) if HAS_NK else False
            
            if abs(max_amp) > abs(min_amp) * 0.5:
                # Strong candidate
                status = "accepted" if nk_agree else "uncertain"
                confidence = 0.9 if nk_agree else 0.6
                landmarks.append({
                    "landmark": "R_peak",
                    "beat_index": i,
                    "x_pixel": int(r_x),
                    "y_pixel": float(-r_y), # Back to image coords
                    "confidence": confidence,
                    "status": status,
                    "detectors_supporting": "raw_morphology,neurokit" if nk_agree else "raw_morphology",
                    "detectors_disagreeing": "none" if nk_agree else "neurokit",
                    "raw_candidate_x": int(r_x),
                    "neurokit_candidate_x": int(x_val[nk_r_peaks[np.argmin([abs((left + local_max_idx) - nk_p) for nk_p in nk_r_peaks])]]) if nk_agree else None,
                    "cwt_candidate_x": None
                })
            else:
                # Weak candidate
                landmarks.append({
                    "landmark": "R_peak",
                    "beat_index": i,
                    "x_pixel": int(r_x),
                    "y_pixel": float(-r_y),
                    "confidence": 0.3,
                    "status": "uncertain",
                    "detectors_supporting": "raw_morphology",
                    "detectors_disagreeing": "neurokit",
                    "raw_candidate_x": int(r_x),
                    "neurokit_candidate_x": None,
                    "cwt_candidate_x": None
                })
                
            # S nadir
            if abs(min_amp) > abs(max_amp) * 0.5:
                landmarks.append({
                    "landmark": "S_nadir",
                    "beat_index": i,
                    "x_pixel": int(s_x),
                    "y_pixel": float(-s_y), # Back to image coords
                    "confidence": 0.9,
                    "status": "accepted",
                    "detectors_supporting": "raw_morphology",
                    "detectors_disagreeing": "none",
                    "raw_candidate_x": int(s_x),
                    "neurokit_candidate_x": None,
                    "cwt_candidate_x": None
                })
                
    df_landmarks = pd.DataFrame(landmarks)
    
    # Package signal processing for plotting
    df_signals = pd.DataFrame({
        "x_pixel": x_val,
        "raw_signal": raw_signal,
        "baseline": baseline,
        "baseline_corrected": baseline_corrected,
        "derivative": derivative,
        "qrs_energy": qrs_energy_smooth
    })
    
    return df_landmarks, df_signals

def run_stage5_2(case):
    print(f"Running Stage 5.2 for {case}...")
    
    stg3_path = f"stage3_output/{case}/09_final_clean_ecg.png"
    stg4_ref_path = f"stage5_output/{case}/01_stage3_reference.png"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    clean_masks_dir = f"stage5_output/{case}/04_stage5_clean_masks"
    
    if not os.path.exists(stg3_path) or not os.path.exists(clean_masks_dir):
        print(f"Skipping {case} - missing inputs.")
        return
        
    img = cv2.imread(stg3_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
        
    with open(rois_path, 'r') as f:
        rois = json.load(f)
        
    out_dir = f"stage5_2_output/{case}"
    dirs = ["07_waveform", "08_landmarks", "09_signal_processing", "10_forensic_visualizations"]
    for d in dirs:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for lead, data in rois.items():
        x1, x2 = data['x_start'], data['x_end']
        y1, y2 = data['y_start'], data['y_end']
        
        mask_path = os.path.join(clean_masks_dir, f"{lead}_clean.png")
        if not os.path.exists(mask_path):
            continue
            
        mask_roi = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_roi = img[y1:y2, x1:x2]
        
        # BRANCH A: Waveform Extraction
        df_waveform_base = extract_branch_a(img_roi, mask_roi)
        
        # MORPHOLOGY FEATURES (derived without modifying Branch A raw_y)
        df_waveform = calculate_morphology_features(df_waveform_base)
        
        df_waveform.to_csv(os.path.join(out_dir, "07_waveform", f"{lead}_waveform.csv"), index=False)
        
        # BRANCH B: Landmark Detection
        df_landmarks, df_signals = detect_branch_b(df_waveform_base)
        if not df_landmarks.empty:
            df_landmarks.to_csv(os.path.join(out_dir, "08_landmarks", f"{lead}_landmarks.csv"), index=False)
        if not df_signals.empty:
            df_signals.to_csv(os.path.join(out_dir, "09_signal_processing", f"{lead}_signals.csv"), index=False)
            
            # Plot signals
            plt.figure(figsize=(10, 4))
            plt.plot(df_signals['x_pixel'], df_signals['baseline_corrected'], label='Corrected Signal')
            plt.plot(df_signals['x_pixel'], df_signals['qrs_energy'], label='QRS Energy', alpha=0.5)
            plt.legend()
            plt.title(f"{lead} Signal Processing")
            plt.savefig(os.path.join(out_dir, "09_signal_processing", f"{lead}_plot.png"))
            plt.close()
            
        # Draw on debug image
        # Green trace
        for _, row in df_waveform.iterrows():
            if pd.notna(row['raw_y_subpixel']):
                px = x1 + int(row['x_pixel'])
                py = y1 + int(row['raw_y_subpixel'])
                cv2.circle(debug_img, (px, py), 1, (0, 255, 0), -1)
                
        # Blue R peaks, Yellow S nadirs
        if not df_landmarks.empty:
            for _, lm in df_landmarks.iterrows():
                px = x1 + int(lm['x_pixel'])
                py = y1 + int(lm['y_pixel'])
                if lm['landmark'] == "R_peak":
                    cv2.circle(debug_img, (px, py), 5, (255, 0, 0), -1)
                elif lm['landmark'] == "S_nadir":
                    cv2.circle(debug_img, (px, py), 5, (0, 255, 255), -1)
                    
    cv2.imwrite(os.path.join(out_dir, "10_forensic_visualizations", "stage5_2_debug.png"), debug_img)
    print(f"Finished {case}.")

def main():
    if len(sys.argv) > 1:
        cases = [sys.argv[1]]
    else:
        cases = [d for d in os.listdir("stage5_output") if os.path.isdir(os.path.join("stage5_output", d)) and d not in ["final_validation_results.csv", "rpeak_fidelity_audit.csv", "hybrid_experiments_results.csv", "candidate_scoring_results.csv"]]
    
    for case in cases:
        run_stage5_2(case)

if __name__ == "__main__":
    main()
