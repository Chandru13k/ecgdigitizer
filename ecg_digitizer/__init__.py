"""
ECG Digitizer: Production Frozen Package (v1.0.0-frozen)
High-Fidelity 7-Stage End-to-End ECG Digitization Pipeline
"""

__version__ = "1.0.0-frozen"
__author__ = "Chandru"

from ecg_digitizer.modules.stage01_pdf_renderer import process_pdf
from ecg_digitizer.modules.stage02_ecg_panel_detection import detect_ecg_panel
from ecg_digitizer.modules.stage02_2_ecg_panel_refinement import refine_ecg_panel
from ecg_digitizer.modules.stage03_grid_removal import run_stage3
from ecg_digitizer.modules.stage03_5_glyph_disentanglement import run_stage3_5_glyph_disentanglement
from ecg_digitizer.modules.stage04_lead_segmentation import run_stage4
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from ecg_digitizer.modules.stage06_wfdb_export import (
    load_stage5_leads,
    get_calibration_factors,
    convert_stage5_to_physical_signals,
    export_wfdb_record,
    perform_roundtrip_validation
)
