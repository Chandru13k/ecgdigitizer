import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def export_digitized_ecg(out_dir_path, base_name, time_axis, raw_leads, clean_leads, reports, overall_status, config):
    out_dir = Path(out_dir_path)
    
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Export CSVs
    raw_df = pd.DataFrame({"time_s": time_axis})
    clean_df = pd.DataFrame({"time_s": time_axis})
    
    max_len = len(time_axis)
    
    for lead in raw_leads.keys():
        s_raw = raw_leads[lead]
        s_clean = clean_leads[lead]
        
        if len(s_raw) < max_len:
            s_raw = np.pad(s_raw, (0, max_len - len(s_raw)), 'edge')
        if len(s_clean) < max_len:
            s_clean = np.pad(s_clean, (0, max_len - len(s_clean)), 'edge')
            
        raw_df[lead] = s_raw[:max_len]
        clean_df[lead] = s_clean[:max_len]
        
    raw_csv = raw_dir / "signals.csv"
    clean_csv = clean_dir / "signals.csv"
    raw_npy = raw_dir / "signals.npy"
    clean_npy = clean_dir / "signals.npy"
        
    raw_df.to_csv(raw_csv, index=False)
    clean_df.to_csv(clean_csv, index=False)
    
    raw_arr = raw_df.to_numpy()
    clean_arr = clean_df.to_numpy()
    np.save(raw_npy, raw_arr)
    np.save(clean_npy, clean_arr)
    
    # Strict Numerical Validation of NPY and CSV
    raw_df_reloaded = pd.read_csv(raw_csv)
    raw_arr_reloaded = np.load(raw_npy)
    
    max_diff = np.max(np.abs(raw_df_reloaded.to_numpy() - raw_arr_reloaded))
    if max_diff >= 1e-5:
        raise ValueError(f"Numerical validation failed between CSV and NPY! max_diff={max_diff}")
    
    metadata = {
        "status": overall_status,
        "csv_npy_validation_max_diff": float(max_diff),
        "reports": reports
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
