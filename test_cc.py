import cv2
import numpy as np
import json

out_dir = "stage3_output/Copy of 20_male"
cand = cv2.imread(f"{out_dir}/07_01_candidate_mask.png", cv2.IMREAD_GRAYSCALE)

# Full CC
num_l, lbls, sts, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)

with open("stage2_2_output/Copy of 20_male/stage2_2_metrics.json", "r") as f:
    metrics = json.load(f)
    
img = cv2.imread("stage2_2_output/Copy of 20_male/01_original.png")
act_y1 = int(metrics.get('active_waveform_y1', 0))
img_h = metrics.get("input_height", 1700)

detected_rows = metrics.get('detected_rows', [])
col_w = cand.shape[1] // 4
r_idx = 0 # Row 1 (V1)
c_idx = 2 # Col 3 (V1)

row_y1 = int(detected_rows[r_idx]["normalized_y1"] * img_h) - act_y1
row_y2 = int(detected_rows[r_idx]["normalized_y2"] * img_h) - act_y1

roi_y1 = max(0, row_y1 + int((row_y2 - row_y1) * 0.4))
roi_y2 = int(detected_rows[r_idx+1]["normalized_y1"] * img_h) - act_y1

col_x1 = c_idx * col_w
roi_x1 = col_x1
roi_x2 = col_x1 + int(col_w * 0.3)

print(f"V1 Label ROI: x({roi_x1}-{roi_x2}), y({roi_y1}-{roi_y2})")

# Let's find the CC that is inside this ROI and corresponds to the tip of V2
roi_lbls = lbls[roi_y1:roi_y2, roi_x1:roi_x2]
unique_labels = np.unique(roi_lbls)
print(f"Labels in ROI: {unique_labels}")

for label in unique_labels:
    if label == 0: continue
    bx, by, bw, bh, barea = sts[label]
    print(f"Label {label}: area={barea}, w={bw}, h={bh}")
