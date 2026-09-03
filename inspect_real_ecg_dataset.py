import os
import json
import glob
import pandas as pd

cases = [d for d in os.listdir("stage5_2_output") if os.path.isdir(os.path.join("stage5_2_output", d)) and "Copy" in d]
cases.sort()

print(f"Total real ECG patient folders found: {len(cases)}")

valid_cases = []
for case in cases:
    panel_path = f"stage3_output/{case}/01_active_panel.png"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    wf_dir = f"stage5_2_output/{case}/07_waveform"

    if os.path.exists(panel_path) and os.path.exists(rois_path) and os.path.exists(wf_dir):
        csv_files = glob.glob(os.path.join(wf_dir, "*.csv"))
        valid_cases.append((case, len(csv_files)))
    else:
        print(f"[!] Missing assets for: {case}")

print(f"Verified {len(valid_cases)} complete real ECG patient records with full lead CSV files:")
for case, n_csv in valid_cases:
    print(f"  - {case}: {n_csv} lead CSV files")
