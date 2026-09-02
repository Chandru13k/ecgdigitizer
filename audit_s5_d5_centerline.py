import os
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_robust_centerline(mask, roi_gray):
    """
    Extracts 1D centerline from binary mask using subpixel grayscale weighting (I^2).
    """
    h, w = mask.shape
    centerline = []
    
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) > 0:
            inverted_gray = 255.0 - col_gray[active_indices]
            weights = (inverted_gray ** 2) + 1e-5
            y_center = np.sum(active_indices * weights) / np.sum(weights)
            width = np.max(active_indices) - np.min(active_indices)
            centerline.append({"x": x, "y": float(y_center), "confidence": 1.0, "width": float(width)})
        else:
            centerline.append({"x": x, "y": None, "confidence": 0.0, "width": 0.0})
            
    df = pd.DataFrame(centerline)
    return df

def audit_s5_d5():
    print("Running S5-D5: Reference Mask vs Predicted Mask Centerline Test...")
    
    case = "Copy of 19_male"
    
    gt_dir = f"pseudo_labels/{case}"
    pred_dir = f"stage5_output/{case}/04_stage5_clean_masks"
    rois_path = f"stage4_output/{case}/lead_rois.json"
    actual_dir = f"stage5_output/{case}/08_actual_leads"
    
    out_dir = f"audit_s5_d5_output/{case}"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print("Missing required directories.")
        return
        
    for file in os.listdir(gt_dir):
        if not file.endswith("_mask.png"):
            continue
            
        lead = file.replace("_mask.png", "")
        
        gt_mask_path = os.path.join(gt_dir, file)
        pred_mask_path = os.path.join(pred_dir, f"{lead}_clean.png")
        actual_path = os.path.join(actual_dir, f"{lead}_actual.png")
        
        if not os.path.exists(pred_mask_path) or not os.path.exists(actual_path):
            continue
            
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        actual_img = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)
        
        # S5-D0 ensure crop size match
        orig_h, orig_w = actual_img.shape
        
        # GT mask is padded to 32
        gt_mask_orig = gt_mask[0:orig_h, 0:orig_w]
        
        # Run centerline on Reference (A)
        df_A = extract_robust_centerline(gt_mask_orig, actual_img)
        
        # Run centerline on Predicted (B)
        df_B = extract_robust_centerline(pred_mask, actual_img)
        
        # Plot A vs B vs Original
        plt.figure(figsize=(15, 6))
        plt.imshow(actual_img, cmap='gray', alpha=0.5)
        plt.plot(df_A['x'], df_A['y'], color='blue', linewidth=1.5, label='Reconstruction A (Reference Mask)')
        plt.plot(df_B['x'], df_B['y'], color='red', linewidth=1.5, label='Reconstruction B (Predicted Mask)')
        
        plt.title(f"S5-D5: {case} - Lead {lead}\nBlue=Centerline from Reference | Red=Centerline from Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{lead}_d5_compare.png"), dpi=150)
        plt.close()
        
        print(f"Generated comparison for {lead}")
        
if __name__ == "__main__":
    audit_s5_d5()
