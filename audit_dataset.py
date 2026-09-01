import cv2
import numpy as np
import json
import os
import glob
import subprocess
import time

def process_file(pdf_path):
    name = os.path.basename(pdf_path).replace('.pdf', '')
    print(f"\n[{name}] Starting processing...")
    
    # Check if Stage 2.2 output exists
    metrics_path = f"stage2_2_output/{name}/stage2_2_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"[{name}] Running Stages 1 and 2...")
        subprocess.run(["python", "run_stage1.py", "--input", pdf_path], check=True)
        s1_out = f"stage1_output/{name}/page_001.png"
        subprocess.run(["python", "run_stage2_1.py", "--input", s1_out], check=True)
        subprocess.run(["python", "run_stage2_2.py", "--stage1_dir", f"stage1_output/{name}", "--stage2_1_dir", f"stage2_1_output/{name}"], check=True)
        
    print(f"[{name}] Running Stage 3...")
    s1_out = f"stage1_output/{name}/page_001.png"
    subprocess.run(["python", "run_stage3.py", "--input", s1_out, "--metrics", metrics_path], check=True)
    
    # Audit logic
    out_dir = f"stage3_output/{name}"
    cand = cv2.imread(f"{out_dir}/07_01_candidate_mask.png", cv2.IMREAD_GRAYSCALE)
    fin = cv2.imread(f"{out_dir}/07_06_final_ecg_mask.png", cv2.IMREAD_GRAYSCALE)
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    img = cv2.imread(f"stage2_2_output/{name}/01_original.png")
    act_y1 = int(metrics.get('active_waveform_y1', 0))
    act_y2 = int(metrics.get('active_waveform_y2', img.shape[0]))
    act_x1 = int(metrics.get('active_x1', 0))
    act_x2 = int(metrics.get('active_x2', img.shape[1]))
    
    active_panel = img[act_y1:act_y2, act_x1:act_x2]
    h, w = active_panel.shape[:2]
    gray = cv2.cvtColor(active_panel, cv2.COLOR_BGR2GRAY)
    _, orig_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    b, g, r = cv2.split(active_panel.astype(np.int16))
    red_dominance = r - np.maximum(g, b)
    is_pure_grid = (red_dominance > 25) & (gray > 140)
    
    orig_px = np.sum(orig_mask > 0)
    cand_px = np.sum(cand > 0)
    fin_px = np.sum(fin > 0)
    rejected_px = orig_px - cand_px
    
    rejected_mask = cv2.bitwise_and(orig_mask, cv2.bitwise_not(cand))
    rejected_red_grid = int(np.sum((rejected_mask > 0) & is_pure_grid))
    rejected_non_red = int(np.sum((rejected_mask > 0) & ~is_pure_grid))
    
    loss_ratio = rejected_non_red / orig_px if orig_px > 0 else 0
    
    print(f"[{name}] Non-red pixel loss ratio: {loss_ratio:.6f}")
    
    if loss_ratio > 0.0001:
        print(f"[{name}] WARNING: Loss exceeded tolerance! Generating diagnostic.")
        m1 = np.zeros((h, w, 3), dtype=np.uint8)
        m1[(orig_mask > 0) & (cand > 0)] = (0, 255, 0)
        m1[(orig_mask > 0) & (cand == 0)] = (0, 0, 255)
        cv2.imwrite(f"C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\{name}_diagnostic.png", m1)
        
    return {
        "name": name,
        "orig_px": int(orig_px),
        "cand_px": int(cand_px),
        "fin_px": int(fin_px),
        "rejected_red_grid": rejected_red_grid,
        "rejected_non_red": rejected_non_red,
        "loss_ratio": loss_ratio
    }

def main():
    pdfs = glob.glob("ECG_D/*.pdf")
    results = []
    # Test all 30!
    for pdf in pdfs: 
        try:
            res = process_file(pdf)
            results.append(res)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            
    print("\n" + "="*50)
    print("STAGE 3 AUDIT SUMMARY")
    print("="*50)
    
    with open("C:\\Users\\Chandru\\.gemini\\antigravity-ide\\brain\\3d7d52fb-94aa-4c9e-9261-4f8f1bc0fdac\\stage3_audit_summary.md", "w") as f:
        f.write("# Stage 3 Dataset Audit Summary\n\n")
        f.write("| Image | Orig Px | Cand Px | Fin Px | Non-Red Loss | Loss Ratio |\n")
        f.write("|-------|---------|---------|--------|--------------|------------|\n")
        
        for r in results:
            print(f"{r['name']}: Loss Ratio = {r['loss_ratio']:.6f} | Non-red Rejected = {r['rejected_non_red']}")
            f.write(f"| {r['name']} | {r['orig_px']} | {r['cand_px']} | {r['fin_px']} | {r['rejected_non_red']} | {r['loss_ratio']:.6f} |\n")

if __name__ == "__main__":
    main()
