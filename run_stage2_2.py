import argparse
import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from ecg_digitizer.modules.stage02_2_ecg_panel_refinement import refine_ecg_panel

def run_stage2_2(stage2_1_metrics_path, img_path):
    print(f"Running Stage 2.2 Panel Refinement on {img_path.name}...")
    
    with open(stage2_1_metrics_path, 'r') as f:
        metrics = json.load(f)
        
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Error: Could not load image at {img_path}")
        return False
        
    try:
        result = refine_ecg_panel(image, metrics)
    except Exception as e:
        print(f"Error refining panel: {e}")
        return False
        
    debug_images = result.pop("debug_images", {})
    
    pdf_name = img_path.parent.name
    out_dir = Path("stage2_2_output") / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stage2_1_bbox = result["stage2_1_bbox"]
    stage2_2_bbox = result["stage2_2_bbox"]
    
    # 01_structural_grid.png (use grid_evidence if we can't easily visualize the grid over the image, or just use 01_full_page.png if we want, but let's just make 01_full_page.png the requested 01_original)
    cv2.imwrite(str(out_dir / "01_original.png"), image)
    
    # Overlay Visualization: Show both bounding boxes
    vis = image.copy()
    
    x1_old, y1_old, x2_old, y2_old = stage2_1_bbox["x1"], stage2_1_bbox["y1"], stage2_1_bbox["x2"], stage2_1_bbox["y2"]
    cv2.rectangle(vis, (x1_old, y1_old), (x2_old, y2_old), (0, 0, 255), 4) # Red = Stage 2.1 (Original)
    
    x1, y1, x2, y2 = stage2_2_bbox["x1"], stage2_2_bbox["y1"], stage2_2_bbox["x2"], stage2_2_bbox["y2"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4) # Green = Stage 2.2 (Refined)
    
    # Add a legend
    cv2.putText(vis, "Stage 2.1 (Original) - Red", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(vis, "Stage 2.2 (Refined) - Green", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(str(out_dir / "02_boundary_comparison.png"), vis)
    
    # 03_ecg_panel_crop.png
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(str(out_dir / "03_ecg_panel_crop.png"), crop)
    
    if "04_row_detection" in debug_images:
        cv2.imwrite(str(out_dir / "04_row_detection.png"), debug_images["04_row_detection"])
        
    if "05_calibration_rejection" in debug_images:
        cv2.imwrite(str(out_dir / "05_calibration_rejection.png"), debug_images["05_calibration_rejection"])
        
    # stage2_2_metrics.json
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(out_dir / "stage2_2_metrics.json", 'w') as f:
        json.dump(result, f, indent=4, default=convert_numpy_types)
        
    print("=" * 40)
    print("STAGE 2.2 — ECG PANEL REFINEMENT")
    print("=" * 40)
    print(f"Input: {img_path}")
    print(f"Stage 2.1 BBox: ({x1_old}, {y1_old}) to ({x2_old}, {y2_old})")
    print(f"Stage 2.2 BBox: ({x1}, {y1}) to ({x2}, {y2})")
    
    d = result["delta"]
    print(f"Deltas: dx1={d['x1']}, dy1={d['y1']}, dx2={d['x2']}, dy2={d['y2']}")
    
    status = result["boundary_status"]
    print("Boundary Status:")
    for side in ["left", "top", "right", "bottom"]:
        print(f"  {side.capitalize()}: {status[side]}")
        
    print("=" * 40)
    print(f"Outputs saved to: {out_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Stage 2.2: ECG Panel Refinement")
    parser.add_argument("--stage2_1_dir", default="stage2_1_output", help="Directory containing Stage 2.1 outputs")
    parser.add_argument("--stage1_dir", default="stage1_output", help="Directory containing original Stage 1 image outputs")
    
    args = parser.parse_args()
    s21_dir = Path(args.stage2_1_dir)
    s1_dir = Path(args.stage1_dir)
    
    if not s21_dir.exists() or not s1_dir.exists():
        print(f"Error: Required input directories not found")
        sys.exit(1)
        
    for pdf_dir in s21_dir.iterdir():
        if pdf_dir.is_dir():
            metrics_path = pdf_dir / "stage2_1_metrics.json"
            img_path = s1_dir / pdf_dir.name / "page_001.png"
            
            if metrics_path.exists() and img_path.exists():
                run_stage2_2(metrics_path, img_path)

if __name__ == "__main__":
    main()
