import os
import cv2
import json
import pandas as pd
import numpy as np

def extract_peaks_valleys(df):
    """
    Very crude peak/valley extractor from centerline dataframe to measure R/S amplitudes.
    """
    valid = df.dropna(subset=['y'])
    if len(valid) == 0:
        return {"r_amp": None, "s_amp": None}
        
    y_min = valid['y'].min() # highest physical point (R peak usually)
    y_max = valid['y'].max() # lowest physical point (S wave usually)
    
    return {"r_amp": float(y_min), "s_amp": float(y_max)}

def audit_s5_morphology():
    print("Running S5-D7/D8: Morphology Audit...")
    
    case = "Copy of 19_male"
    lead = "V2" # Focus on the difficult case as per user request
    
    gt_dir = f"pseudo_labels/{case}"
    pred_dir = f"stage5_output/{case}/04_stage5_clean_masks"
    actual_dir = f"stage5_output/{case}/08_actual_leads"
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print("Missing required directories.")
        return
        
    # We will use the centerlines generated in S5-D5
    d5_dir = f"audit_s5_d5_output/{case}"
    
    # Actually, we didn't save the raw CSVs in S5-D5, we just plotted them.
    # Let's extract the centerlines again for morphology math.
    from audit_s5_d5_centerline import extract_robust_centerline
    
    gt_mask_path = os.path.join(gt_dir, f"{lead}_mask.png")
    pred_mask_path = os.path.join(pred_dir, f"{lead}_clean.png")
    actual_path = os.path.join(actual_dir, f"{lead}_actual.png")
    
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    actual_img = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)
    
    orig_h, orig_w = actual_img.shape
    gt_mask_orig = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    df_A = extract_robust_centerline(gt_mask_orig, actual_img)
    df_B = extract_robust_centerline(pred_mask, actual_img)
    
    morph_A = extract_peaks_valleys(df_A)
    morph_B = extract_peaks_valleys(df_B)
    
    print("\n--- Clinical Morphology Audit for V2 ---")
    print(f"Reference Reconstruction (A) R-Peak Y (min): {morph_A['r_amp']:.1f}")
    print(f"Predicted Reconstruction (B) R-Peak Y (min): {morph_B['r_amp']:.1f}")
    print(f"Reference Reconstruction (A) S-Depth Y (max): {morph_A['s_amp']:.1f}")
    print(f"Predicted Reconstruction (B) S-Depth Y (max): {morph_B['s_amp']:.1f}")
    
    delta_r = abs(morph_A['r_amp'] - morph_B['r_amp'])
    delta_s = abs(morph_A['s_amp'] - morph_B['s_amp'])
    print(f"\nΔR Amplitude (Pixels): {delta_r:.1f}")
    print(f"ΔS Depth (Pixels): {delta_s:.1f}")
    
    with open("s5_morphology_results.json", "w") as f:
        json.dump({
            "Reference_R": morph_A['r_amp'],
            "Predicted_R": morph_B['r_amp'],
            "Delta_R": delta_r,
            "Reference_S": morph_A['s_amp'],
            "Predicted_S": morph_B['s_amp'],
            "Delta_S": delta_s
        }, f, indent=4)
        
if __name__ == "__main__":
    audit_s5_morphology()
