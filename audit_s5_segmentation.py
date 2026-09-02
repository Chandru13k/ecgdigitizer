import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18

def calculate_metrics(preds, targets):
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    return float(precision), float(recall), float(iou), float(dice)

def audit_s5_segmentation():
    print("Running S5-D1 to S5-D4: Segmentation and Threshold Sweep...")
    
    cases = ["Copy of 20_male"] # Validation case
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device))
    model.eval()
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for case in cases:
        gt_dir = f"pseudo_labels/{case}"
        actual_dir = f"stage5_output/{case}/08_actual_leads"
        
        out_dir = f"audit_s5_segmentation_output/{case}"
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.exists(gt_dir) or not os.path.exists(actual_dir):
            continue
            
        for file in os.listdir(gt_dir):
            if not file.endswith("_mask.png"):
                continue
                
            lead = file.replace("_mask.png", "")
            
            gt_mask = cv2.imread(os.path.join(gt_dir, file), cv2.IMREAD_GRAYSCALE)
            actual_img = cv2.imread(os.path.join(actual_dir, f"{lead}_actual.png"), cv2.IMREAD_GRAYSCALE)
            
            orig_h, orig_w = actual_img.shape
            gt_mask_resized = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_mask_orig = gt_mask_resized
            
            pad_h = (32 - (orig_h % 32)) % 32
            pad_w = (32 - (orig_w % 32)) % 32
            roi_padded = cv2.copyMakeBorder(actual_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            
            tensor_img = torch.from_numpy(roi_padded).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor_img = tensor_img.to(device)
            
            with torch.no_grad():
                output = model(tensor_img)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                
            prob_orig = prob[0:orig_h, 0:orig_w]
            
            # S5-D1/D2: Save probability map
            prob_map = (prob_orig * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{lead}_prob_map.png"), prob_map)
            
            gt_tensor = torch.from_numpy(gt_mask_orig).float() / 255.0
            prob_tensor = torch.from_numpy(prob_orig).float()
            
            for t in thresholds:
                preds = (prob_tensor > t).float()
                p, r, i, d = calculate_metrics(preds, gt_tensor)
                
                results.append({
                    "case": case,
                    "lead": lead,
                    "threshold": t,
                    "precision": p,
                    "recall": r,
                    "iou": i,
                    "dice": d
                })
                
                # S5-D4: Connected components for V2 at optimal threshold (assume 0.4 for now)
                if lead == "V2" and t == 0.4:
                    binary = (prob_orig > t).astype(np.uint8) * 255
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    cc_out = np.zeros_like(binary)
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] >= 10:
                            cc_out[labels == i] = 255
                            
                    cv2.imwrite(os.path.join(out_dir, f"{lead}_cc_mask_t0.4.png"), cc_out)
                    
    df = pd.DataFrame(results)
    df.to_csv("s5_threshold_sweep.csv", index=False)
    
    # Aggregate results
    agg = df.groupby("threshold")[["precision", "recall", "iou", "dice"]].mean().reset_index()
    print("--- Full Validation Set Average ---")
    print(agg)
    
    v2 = df[df["lead"] == "V2"].groupby("threshold")[["precision", "recall", "iou", "dice"]].mean().reset_index()
    print("\n--- V2 Lead Only ---")
    print(v2)

if __name__ == "__main__":
    audit_s5_segmentation()
