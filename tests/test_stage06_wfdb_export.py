import os
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
import wfdb

from ecg_digitizer.modules.stage06_wfdb_export import (
    convert_stage5_to_physical_signals,
    export_wfdb_record,
    perform_roundtrip_validation,
    LEAD_ORDER,
    MAX_QUANT_ERR_MV,
    REF_FS,
    REF_GAIN
)

def test_stage06_wfdb_export_and_roundtrip():
    # 1. Create synthetic lead coordinates (10 seconds at 500 Hz = 5000 points)
    duration_sec = 2.5
    fs_px = 300.0  # px per second
    px_per_mv = 120.0  # px per mV
    num_pts = int(duration_sec * fs_px)
    
    leads_data = {}
    x_coords = np.arange(num_pts, dtype=float)
    
    for lead in LEAD_ORDER:
        # Generate 1 Hz sine wave with 1 mV peak
        y_coords = 200.0 - (px_per_mv * np.sin(2 * np.pi * 1.0 * (x_coords / fs_px)))
        leads_data[lead] = (x_coords, y_coords)
        
    # 2. Convert to physical matrix
    t_grid, physical_matrix, calibrated_signals = convert_stage5_to_physical_signals(
        leads_data, px_per_sec=fs_px, px_per_mv=px_per_mv, target_fs=REF_FS
    )
    
    assert physical_matrix.shape[1] == 12, "Should export 12 standard leads"
    assert physical_matrix.shape[0] == len(t_grid)
    
    # 3. Export to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        record_name = "test_ecg_rec"
        
        hea_path, dat_path, sanitized_name = export_wfdb_record(
            out_path, record_name, physical_matrix, fs=REF_FS
        )
        
        assert hea_path.exists(), "WFDB header .hea not created"
        assert dat_path.exists(), "WFDB binary .dat not created"
        
        # 4. Perform roundtrip validation
        status, metrics, checks, read_signal = perform_roundtrip_validation(
            hea_path, physical_matrix, sanitized_name
        )
        
        assert status == "PASS", f"WFDB Round-trip failed: {checks}"
        assert checks["fs_correct"] is True
        assert checks["n_sig_correct"] is True
        assert checks["sig_name_correct"] is True
        assert checks["units_correct"] is True
        assert checks["adc_gain_correct"] is True
        
        # Correlation across all leads should be > 0.999
        for m in metrics:
            assert m["correlation"] > 0.999
            assert m["max_abs_error_mv"] <= MAX_QUANT_ERR_MV + 1e-4
