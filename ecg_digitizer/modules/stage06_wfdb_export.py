import os
import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path

# Verified reference parameters from reference_wfdb/01000_lr.hea & .dat (and referennce_wfbd/01000_lr)
REF_FS = 100  # Sampling frequency from reference record
REF_FMT = '16'  # 16-bit signed integer format
REF_GAIN = 1000.0  # 1000 LSB / mV
REF_BASELINE = 0
REF_UNITS = 'mV'
REF_ADC_RES = 16
REF_ADC_ZERO = 0
LEAD_ORDER = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

MAX_QUANT_ERR_MV = 0.5 / REF_GAIN  # 0.0005 mV theoretical max rounding error for 1000 LSB/mV

def get_calibration_factors(case_dir):
    """
    Extract px_per_mm_x and px_per_mm_y from stage metadata if available,
    otherwise fallback to standard 12.0 px/mm.
    """
    px_per_mm_x = 12.0
    px_per_mm_y = 12.0
    
    stage4_meta = Path(case_dir).parent.parent / "stage4_output" / Path(case_dir).name / "stage4_metrics.json"
    if stage4_meta.exists():
        try:
            with open(stage4_meta, 'r') as f:
                data = json.load(f)
                if 'px_per_mm_x' in data: px_per_mm_x = float(data['px_per_mm_x'])
                if 'px_per_mm_y' in data: px_per_mm_y = float(data['px_per_mm_y'])
        except Exception:
            pass
            
    px_per_sec = 25.0 * px_per_mm_x
    px_per_mv = 10.0 * px_per_mm_y
    return px_per_sec, px_per_mv, px_per_mm_x, px_per_mm_y

def load_stage5_leads(case_dir):
    """
    Load digitized centerline CSVs from Stage 05 output.
    Returns dictionary mapping lead_name -> (x_coords, y_coords).
    """
    traces_dir = Path(case_dir) / "06_digitized_traces"
    leads_data = {}
    
    # Lead file mapping
    file_map = {
        'I': 'I_centerline.csv',
        'II': 'II_centerline.csv',
        'III': 'III_centerline.csv',
        'AVR': 'aVR_centerline.csv',
        'AVL': 'aVL_centerline.csv',
        'AVF': 'aVF_centerline.csv',
        'V1': 'V1_centerline.csv',
        'V2': 'V2_centerline.csv',
        'V3': 'V3_centerline.csv',
        'V4': 'V4_centerline.csv',
        'V5': 'V5_centerline.csv',
        'V6': 'V6_centerline.csv'
    }
    
    for lead_name in LEAD_ORDER:
        fname = file_map[lead_name]
        fpath = traces_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing Stage 05 centerline file for lead {lead_name}: {fpath}")
            
        df = pd.read_csv(fpath)
        # Ensure column names are present
        if 'y' in df.columns:
            y_vals = df['y'].values
        elif 'raw_y_subpixel' in df.columns:
            y_vals = df['raw_y_subpixel'].values
        else:
            y_vals = df.iloc[:, 1].values
            
        x_vals = df['x'].values if 'x' in df.columns else df.iloc[:, 0].values
        leads_data[lead_name] = (x_vals.astype(float), y_vals.astype(float))
        
    return leads_data

def convert_stage5_to_physical_signals(leads_data, px_per_sec, px_per_mv, target_fs=REF_FS):
    """
    Convert Stage 05 pixel coordinates to calibrated physical signals (mV)
    sampled at target_fs, maintaining exact waveform morphology.
    """
    calibrated_signals = {}
    
    for lead_name, (xs, ys) in leads_data.items():
        # Determine baseline y as median of valid y values
        valid_mask = ~np.isnan(ys)
        if not np.any(valid_mask):
            raise ValueError(f"Lead {lead_name} has no valid y coordinates!")
            
        y_baseline = np.median(ys[valid_mask])
        
        # Convert y to mV: smaller y is higher excursion on image
        v_mv = (y_baseline - ys) / px_per_mv
        
        # Time axis in seconds
        t_sec = xs / px_per_sec
        
        calibrated_signals[lead_name] = (t_sec, v_mv)
        
    # Determine common time span across all 12 leads
    t_start = max([t[0][0] for t in calibrated_signals.values()])
    t_end = min([t[0][-1] for t in calibrated_signals.values()])
    
    if t_end <= t_start:
        # Fallback to absolute min/max across leads
        t_start = min([t[0][0] for t in calibrated_signals.values()])
        t_end = max([t[0][-1] for t in calibrated_signals.values()])
        
    # Standard grid at target_fs
    num_samples = int(np.round((t_end - t_start) * target_fs)) + 1
    t_grid = np.linspace(t_start, t_end, num_samples)
    
    physical_matrix = np.zeros((num_samples, 12), dtype=np.float64)
    
    for idx, lead_name in enumerate(LEAD_ORDER):
        t_orig, v_orig = calibrated_signals[lead_name]
        valid_mask = ~np.isnan(v_orig)
        
        if np.sum(valid_mask) > 1:
            # Interpolate strictly valid points to target grid
            v_interp = np.interp(t_grid, t_orig[valid_mask], v_orig[valid_mask])
        else:
            v_interp = np.zeros(num_samples)
            
        physical_matrix[:, idx] = v_interp
        
    return t_grid, physical_matrix, calibrated_signals

def export_wfdb_record(record_dir, record_name, physical_matrix, fs=REF_FS):
    """
    Writes WFDB .hea + .dat files matching verified reference specifications.
    """
    out_path = Path(record_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # WFDB record_name cannot contain spaces in .hea header lines
    sanitized_record_name = record_name.replace(" ", "_")
    
    wfdb.wrsamp(
        record_name=sanitized_record_name,
        fs=fs,
        units=[REF_UNITS] * 12,
        sig_name=LEAD_ORDER,
        p_signal=physical_matrix,
        fmt=[REF_FMT] * 12,
        adc_gain=[REF_GAIN] * 12,
        baseline=[REF_BASELINE] * 12,
        write_dir=str(out_path)
    )
    
    dat_path = out_path / f"{sanitized_record_name}.dat"
    hea_path = out_path / f"{sanitized_record_name}.hea"
    
    if not dat_path.exists() or not hea_path.exists():
        raise FileNotFoundError(f"WFDB export failed to produce {dat_path} or {hea_path}")
        
    return hea_path, dat_path, sanitized_record_name

def perform_roundtrip_validation(hea_path, source_physical_matrix, record_name):
    """
    Reads back the exported WFDB record using wfdb.rdrecord()
    and compares read-back signal against source physical signal.
    """
    rec_path = str(Path(hea_path).with_suffix(''))
    read_rec = wfdb.rdrecord(rec_path)
    read_signal = read_rec.p_signal
    
    n_samples, n_leads = source_physical_matrix.shape
    metrics = []
    
    overall_pass = True
    
    for idx, lead in enumerate(LEAD_ORDER):
        src_v = source_physical_matrix[:, idx]
        read_v = read_signal[:, idx]
        
        diff = read_v - src_v
        max_abs_err = np.max(np.abs(diff))
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # Pearson correlation
        if np.std(src_v) > 1e-8 and np.std(read_v) > 1e-8:
            corr = np.corrcoef(src_v, read_v)[0, 1]
        else:
            corr = 1.0
            
        amp_range = [float(np.min(read_v)), float(np.max(read_v))]
        
        # Pass check against maximum quantization error (0.0005 mV)
        lead_pass = max_abs_err <= (MAX_QUANT_ERR_MV + 1e-6)
        if not lead_pass:
            overall_pass = False
            
        metrics.append({
            "lead": lead,
            "sample_count": n_samples,
            "max_abs_error_mv": float(max_abs_err),
            "mae_mv": float(mae),
            "rmse_mv": float(rmse),
            "correlation": float(corr),
            "min_amplitude_mv": amp_range[0],
            "max_amplitude_mv": amp_range[1],
            "pass": bool(lead_pass)
        })
        
    structural_checks = {
        "record_name_matches": read_rec.record_name == record_name,
        "n_sig_correct": read_rec.n_sig == 12,
        "sig_name_correct": read_rec.sig_name == LEAD_ORDER,
        "fs_correct": read_rec.fs == REF_FS,
        "units_correct": read_rec.units == [REF_UNITS] * 12,
        "adc_gain_correct": read_rec.adc_gain == [REF_GAIN] * 12,
        "baseline_correct": read_rec.baseline == [REF_BASELINE] * 12,
        "fmt_correct": read_rec.fmt == [REF_FMT] * 12,
        "dat_file_size_bytes": os.path.getsize(Path(hea_path).parent / f"{record_name}.dat"),
        "expected_dat_size_bytes": n_samples * 12 * 2  # 2 bytes per int16 sample
    }
    
    structural_pass = all([
        structural_checks["record_name_matches"],
        structural_checks["n_sig_correct"],
        structural_checks["sig_name_correct"],
        structural_checks["fs_correct"],
        structural_checks["units_correct"],
        structural_checks["adc_gain_correct"],
        structural_checks["baseline_correct"],
        structural_checks["fmt_correct"],
        structural_checks["dat_file_size_bytes"] == structural_checks["expected_dat_size_bytes"]
    ])
    
    final_status = "PASS" if (overall_pass and structural_pass) else "FAIL"
    
    return final_status, metrics, structural_checks, read_signal
