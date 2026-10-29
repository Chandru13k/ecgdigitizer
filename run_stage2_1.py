import argparse
import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from ecg_digitizer.modules.stage02_ecg_panel_detection import detect_ecg_panel

def run_stage2_1(img_path):
    print(f"Running Stage 2.1 Panel Detection on {img_path.name}...")
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not load image at {img_path}")
        return False
        
    h, w = image.shape[:2]
    
    result = detect_ecg_panel(image)
    
    debug_images = result.pop("debug_images", {})
    
    metrics = {
        "input_image": str(img_path),
        "image_width": w,
        "image_height": h,
        **result
    }
    
    pdf_name = img_path.parent.name
    out_dir = Path("stage2_1_output") / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 01_full_page.png
    cv2.imwrite(str(out_dir / "01_full_page.png"), image)
    
    # 02_grid_evidence.png
    if "grid_evidence" in debug_images:
        cv2.imwrite(str(out_dir / "02_grid_evidence.png"), debug_images["grid_evidence"])
        
    # 03_candidate_regions.png
    candidate_vis = cv2.cvtColor(debug_images.get("grid_evidence", np.zeros((h, w), dtype=np.uint8)), cv2.COLOR_GRAY2BGR)
    bbox = metrics["ecg_panel_bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cv2.rectangle(candidate_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "03_candidate_regions.png"), candidate_vis)
    
    # 04_selected_ecg_panel.png
    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.imwrite(str(out_dir / "04_selected_ecg_panel.png"), vis)
    
    # 05_ecg_panel_crop.png
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(out_dir / "05_ecg_panel_crop.png"), crop)
    
    # stage2_1_metrics.json
    with open(out_dir / "stage2_1_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("=" * 40)
    print("STAGE 2.1 — ECG PANEL DETECTION")
    print("=" * 40)
    print(f"Input: {img_path}")
    print(f"BBox: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Panel dimensions: {metrics['ecg_panel_width']} x {metrics['ecg_panel_height']}")
    print(f"Detection Score: {metrics['overall_detection_score']:.4f}")
    print(f"Status: {metrics['status']}")
    print("=" * 40)
    print(f"Outputs saved to: {out_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Stage 2.1: Automatic ECG Panel Detection")
    parser.add_argument("--input", required=True, help="Path to the Stage 1 rendered PNG or directory containing Stage 1 outputs")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path not found at {input_path}")
        sys.exit(1)
        
    if input_path.is_dir():
        # Find all page_001.png files
        for pdf_dir in input_path.iterdir():
            if pdf_dir.is_dir():
                img_path = pdf_dir / "page_001.png"
                if img_path.exists():
                    run_stage2_1(img_path)
    else:
        run_stage2_1(input_path)

if __name__ == "__main__":
    main()
