import pytest
import cv2
import json
from pathlib import Path
from ecg_digitizer.modules.stage02_ecg_panel_detection import detect_ecg_panel

BASE_DIR = Path(r"C:\Users\Chandru\OneDrive\Desktop\6th sem\trial")
STAGE1_DIR = BASE_DIR / "stage1_output"

def validate_detection_metrics(metrics, h, w):
    bbox = metrics["ecg_panel_bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    assert x1 < x2, "Invalid bounding box width"
    assert y1 < y2, "Invalid bounding box height"
    assert x1 >= 0 and y1 >= 0, "Bounding box exceeds left/top boundaries"
    assert x2 <= w and y2 <= h, "Bounding box exceeds right/bottom boundaries"
    assert metrics["ecg_panel_width"] > 0
    assert metrics["ecg_panel_height"] > 0
    assert metrics["status"] != "FAIL", "Detection failed"
    assert metrics["panel_completeness_check"] == "PASS"
    
    # 2.2 Hardening Checks
    # The bounding box should remain reasonably aligned with the grid unless a massive excursion occurred
    for side in ["left", "right", "top", "bottom"]:
        assert metrics["grid_error"][side] >= 0.0, f"Grid error missing for {side}"
        assert metrics["boundary_margin"][side] >= 0.0, f"Margin missing for {side}"
        assert metrics["expansion_px"][side] >= 0, f"Expansion px missing for {side}"
        
    # Extraneous ink should be minimal
    assert metrics["extraneous_ink_ratio"] < 0.5, "Excessive non-ECG content captured during expansion"
    
    # Check clipping metric flags
    assert metrics["waveform_clipped"] == False
    assert metrics["grid_clipped"] == False
    assert metrics["calibration_clipped"] == False
    assert metrics["panel_tightness_score"] >= 0.0

def test_stage02_panel_detection_19_male():
    img_path = STAGE1_DIR / "Copy of 19_male" / "page_001.png"
    if not img_path.exists():
        pytest.skip(f"Stage 1 image not found at {img_path}")
        
    image = cv2.imread(str(img_path))
    assert image is not None, "Failed to read RGB image"
    assert len(image.shape) == 3 and image.shape[2] == 3, "Image is not RGB"
    h, w = image.shape[:2]
    
    metrics = detect_ecg_panel(image)
    validate_detection_metrics(metrics, h, w)

def test_stage02_panel_detection_20_male():
    img_path = STAGE1_DIR / "Copy of 20_male" / "page_001.png"
    if not img_path.exists():
        pytest.skip(f"Stage 1 image not found at {img_path}")
        
    image = cv2.imread(str(img_path))
    assert image is not None, "Failed to read RGB image"
    assert len(image.shape) == 3 and image.shape[2] == 3, "Image is not RGB"
    h, w = image.shape[:2]
    
    metrics = detect_ecg_panel(image)
    validate_detection_metrics(metrics, h, w)
