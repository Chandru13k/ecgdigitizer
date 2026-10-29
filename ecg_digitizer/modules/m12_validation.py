import numpy as np
import neurokit2 as nk

def validate_reconstruction(volts_mv_raw, volts_mv_clean, times, metrics, config):
    fs = config.get('sampling_rate', 250.0)
    
    continuity = metrics['continuity']
    interpolated = metrics['interpolated_ratio']
    
    # Strict Quality Gate
    min_continuity = config.get('min_continuity', 0.50)
    max_interpolation = config.get('max_interpolation', 0.50)
    
    if continuity < min_continuity or interpolated > max_interpolation:
        return {
            "status": "FAIL",
            "confidence": float(continuity),
            "digitized_hr": "INVALID",
            "num_beats": "INVALID",
            "metrics": metrics,
            "reason": f"Failed quality gate: continuity={continuity:.2f}, interpolation={interpolated:.2f}"
        }
    
    try:
        rpeaks, _ = nk.ecg_peaks(volts_mv_clean, sampling_rate=int(fs))
        num_beats = len(rpeaks['ECG_R_Peaks'])
        
        # Calculate HR
        if num_beats > 1:
            rpeaks_sec = rpeaks['ECG_R_Peaks'] / fs
            rr_intervals = np.diff(rpeaks_sec)
            hr = 60.0 / np.mean(rr_intervals)
        else:
            hr = 0.0
    except:
        hr = 0.0
        num_beats = 0
        
    status = "PASS"
    if continuity < config.get('min_confidence_pass', 0.85):
        status = "WARNING"
        
    return {
        "status": status,
        "confidence": float(continuity),
        "digitized_hr": float(hr) if hr > 0 else "INVALID",
        "num_beats": num_beats,
        "metrics": metrics
    }
