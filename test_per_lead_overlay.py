import os
import json
import cv2
import pandas as pd
import numpy as np

sample_case = "Copy of 19_male"
panel_path = f"stage3_output/{sample_case}/01_active_panel.png"
rois_json_path = f"stage4_output/{sample_case}/lead_rois.json"
waveform_dir = f"stage5_2_output/{sample_case}/07_waveform"

img_panel = cv2.imread(panel_path)
with open(rois_json_path, "r") as f:
    rois = json.load(f)

img_overlay = img_panel.copy()

print(f"Mapping 12 leads + Rhythm II extracted waveforms onto active panel ({img_panel.shape[1]}x{img_panel.shape[0]})...")

for lead_name, roi in rois.items():
    csv_file = os.path.join(waveform_dir, f"{lead_name}_waveform.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping missing lead: {lead_name}")
        continue

    df = pd.read_csv(csv_file)
    x_start = roi["x_start"]
    y_start = roi["y_start"]

    # Filter valid rows
    valid_df = df[df["valid"] == True]
    if len(valid_df) == 0:
        continue

    x_coords = x_start + valid_df["x_pixel"].values
    y_coords = y_start + valid_df["raw_y_subpixel"].values

    pts = np.column_stack((x_coords, y_coords)).astype(np.int32)
    # Draw thick bright lime green line directly over the extracted waves
    cv2.polylines(img_overlay, [pts], isClosed=False, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    # Draw lead bounding box outline and name label
    cv2.rectangle(img_overlay, (x_start, y_start), (roi["x_end"], roi["y_end"]), (255, 0, 0), 2)
    cv2.putText(img_overlay, lead_name, (x_start + 15, y_start + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

out_file = "test_lead_waves_overlay.png"
cv2.imwrite(out_file, img_overlay)
print(f"Saved exact per-lead wave overlay to: {out_file}")
