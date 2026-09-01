import pytest
import cv2
import numpy as np
import json
from pathlib import Path
from ecg_digitizer.modules.stage02_2_ecg_panel_refinement import refine_ecg_panel

STAGE1_DIR = Path("stage1_output")
STAGE2_1_DIR = Path("stage2_1_output")

@pytest.fixture
def sample_19_male():
    img_path = STAGE1_DIR / "Copy of 19_male" / "page_001.png"
    json_path = STAGE2_1_DIR / "Copy of 19_male" / "stage2_1_metrics.json"
    if not img_path.exists() or not json_path.exists():
        pytest.skip("Test files not found")
    
    image = cv2.imread(str(img_path))
    with open(json_path, 'r') as f:
        metrics = json.load(f)
        
    return image, metrics

def test_pipeline_integrity_wrong_stage(sample_19_male):
    image, metrics = sample_19_male
    metrics["stage"] = "stage1" # Wrong stage
    
    with pytest.raises(ValueError, match="PIPELINE INTEGRITY FAILURE"):
        refine_ecg_panel(image, metrics)

def test_pipeline_integrity_wrong_hash(sample_19_male):
    image, metrics = sample_19_male
    metrics["source_sha256"] = "invalid_hash"
    
    with pytest.raises(ValueError, match="PIPELINE INTEGRITY FAILURE"):
        refine_ecg_panel(image, metrics)
        
def test_pipeline_integrity_invalid_coords(sample_19_male):
    image, metrics = sample_19_male
    metrics["ecg_panel_bbox"]["x1"] = 5000 # Out of bounds
    
    with pytest.raises(ValueError, match="PIPELINE INTEGRITY FAILURE"):
        refine_ecg_panel(image, metrics)

def test_directional_refinement(sample_19_male):
    image, metrics = sample_19_male
    
    # Store initial bbox
    initial_bbox = metrics["ecg_panel_bbox"].copy()
    
    result = refine_ecg_panel(image, metrics)
    
    # The output MUST contain exactly the Stage 2.1 bbox it started from
    assert result["stage2_1_bbox"] == initial_bbox
    
    # Verify that it didn't just rerun Stage 1 independently
    assert "stage2_2_bbox" in result
    
    # In 19_male, we expect tightening on all sides because the Stage 2.1 pad_x/pad_y is removed,
    # and no waveforms exceed the grid.
    status = result["boundary_status"]
    assert status["left"] == "tightened"
    assert status["top"] == "tightened"
    assert status["right"] == "tightened"
    assert status["bottom"] == "tightened"

def test_large_excursion_20_male():
    img_path = STAGE1_DIR / "Copy of 20_male" / "page_001.png"
    json_path = STAGE2_1_DIR / "Copy of 20_male" / "stage2_1_metrics.json"
    if not img_path.exists() or not json_path.exists():
        pytest.skip("Test files not found")
        
    image = cv2.imread(str(img_path))
    with open(json_path, 'r') as f:
        metrics = json.load(f)
        
    result = refine_ecg_panel(image, metrics)    
    bbox = result["stage2_2_bbox"]
    assert bbox["x1"] < bbox["x2"]
    assert bbox["y1"] < bbox["y2"]
    
    # Assert structural bounds are reported
    assert "structural_grid_bbox" in result
    assert "active_ecg_panel_bbox" in result
    assert "detected_row_count" in result
    assert result["detected_row_count"] == 4
    
    # The bottom boundary should encompass the final ECG row (Lead II) and exclude footer stats.
    # The true structural grid ends at ~2076, while footer stats start at ~2105.
    # The active ECG panel mathematically ends at ~1876 (bottom of Lead II).
    assert bbox["y2"] < 2100, "Footer text was included in the crop!"
    assert bbox["y2"] > 1850, "Lead II row was likely clipped!"

def test_resize_invariance(sample_19_male):
    image, metrics = sample_19_male
    h, w = image.shape[:2]
    
    # Run original
    result_orig = refine_ecg_panel(image, metrics)
    norm_orig = result_orig["normalized_bbox"]
    
    scales = [(0.5, 0.5), (1.5, 1.5), (2.0, 2.0)]
    
    for sx, sy in scales:
        new_w = int(w * sx)
        new_h = int(h * sy)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_CUBIC)
        
        # We must create a new fake Stage 2.1 JSON for the resized image
        # since Stage 2.1 would have output appropriately scaled coordinates.
        new_metrics = metrics.copy()
        new_bbox = new_metrics["ecg_panel_bbox"].copy()
        new_bbox["x1"] = int(new_bbox["x1"] * sx)
        new_bbox["x2"] = int(new_bbox["x2"] * sx)
        new_bbox["y1"] = int(new_bbox["y1"] * sy)
        new_bbox["y2"] = int(new_bbox["y2"] * sy)
        new_metrics["ecg_panel_bbox"] = new_bbox
        
        # Update hash to match the new image
        import hashlib
        new_metrics["source_sha256"] = hashlib.sha256(resized_img.tobytes()).hexdigest()
        
        result_resized = refine_ecg_panel(resized_img, new_metrics)
        norm_resized = result_resized["normalized_bbox"]
        
        # Assert normalized bounds are approximately invariant (within 3.5% tolerance)
        assert abs(norm_orig["x1_norm"] - norm_resized["x1_norm"]) < 0.035
        assert abs(norm_orig["y1_norm"] - norm_resized["y1_norm"]) < 0.035
        assert abs(norm_orig["x2_norm"] - norm_resized["x2_norm"]) < 0.035
        assert abs(norm_orig["y2_norm"] - norm_resized["y2_norm"]) < 0.035

def test_non_uniform_resize(sample_19_male):
    image, metrics = sample_19_male
    h, w = image.shape[:2]
    
    result_orig = refine_ecg_panel(image, metrics)
    norm_orig = result_orig["normalized_bbox"]
    
    # Distort image
    sx, sy = 0.75, 1.25
    new_w = int(w * sx)
    new_h = int(h * sy)
    resized_img = cv2.resize(image, (new_w, new_h))
    
    new_metrics = metrics.copy()
    new_bbox = new_metrics["ecg_panel_bbox"].copy()
    new_bbox["x1"] = int(new_bbox["x1"] * sx)
    new_bbox["x2"] = int(new_bbox["x2"] * sx)
    new_bbox["y1"] = int(new_bbox["y1"] * sy)
    new_bbox["y2"] = int(new_bbox["y2"] * sy)
    new_metrics["ecg_panel_bbox"] = new_bbox
    
    import hashlib
    new_metrics["source_sha256"] = hashlib.sha256(resized_img.tobytes()).hexdigest()
    
    result_resized = refine_ecg_panel(resized_img, new_metrics)
    norm_resized = result_resized["normalized_bbox"]
    
    # Assert scale independence even with distortion
    assert abs(norm_orig["x1_norm"] - norm_resized["x1_norm"]) < 0.035
    assert abs(norm_orig["y1_norm"] - norm_resized["y1_norm"]) < 0.035
    assert abs(norm_orig["x2_norm"] - norm_resized["x2_norm"]) < 0.035
    assert abs(norm_orig["y2_norm"] - norm_resized["y2_norm"]) < 0.035

def test_synthetic_calibration_and_excursion(sample_19_male):
    image, metrics = sample_19_male
    
    # We will draw a mock calibration pulse and a mock extreme QRS excursion
    # onto the image in the footer area (where it won't connect to existing traces).
    test_img = image.copy()
    
    # Run original first to get grid spacing
    result_orig = refine_ecg_panel(image, metrics)
    gs_x = result_orig["detected_grid_spacing_x"]
    gs_y = result_orig["detected_grid_spacing_y"]
    
    # Draw a mock calibration pulse (compact, blocky, high fill)
    # e.g., 3 * gs_x width, 10 * gs_y height
    cal_x = 500
    cal_y = 1900
    cal_w = int(gs_x * 3)
    cal_h = int(gs_y * 10)
    
    # Make it a solid block (high fill ratio)
    cv2.rectangle(test_img, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), (0, 0, 0), -1)
    
    # Draw a mock extreme QRS excursion (very tall, very thin, low fill)
    # e.g., 1 * gs_x width, 25 * gs_y height
    qrs_x = 1000
    qrs_y = 1900
    qrs_w = int(gs_x * 1)
    qrs_h = int(gs_y * 25)
    
    # Make it a thin vertical line (low fill ratio)
    cv2.line(test_img, (qrs_x, qrs_y + qrs_h), (qrs_x, qrs_y), (0, 0, 0), 2)
    
    import hashlib
    new_metrics = metrics.copy()
    new_metrics["source_sha256"] = hashlib.sha256(test_img.tobytes()).hexdigest()
    
    # Ensure it's not excluded by the Stage 2.1 bounds
    new_metrics["ecg_panel_bbox"]["y2"] = test_img.shape[0] - 1
    
    result = refine_ecg_panel(test_img, new_metrics)
    
    cal_bboxes = result.get("rejected_calibration_components", [])
    
    # Check if the calibration block was correctly rejected
    found_cal = False
    for bx, by, bw, bh in cal_bboxes:
        if abs(bx - cal_x) < 5 and abs(by - cal_y) < 5:
            found_cal = True
            break
            
    assert found_cal, "Synthetic calibration pulse was NOT rejected as calibration!"
    
    # Check if the QRS excursion was INCORRECTLY rejected as calibration
    found_qrs_in_cal = False
    for bx, by, bw, bh in cal_bboxes:
        if abs(bx - qrs_x) < 5 and abs(by - qrs_y) < 5:
            found_qrs_in_cal = True
            break
            
    assert not found_qrs_in_cal, "Synthetic extreme QRS was INCORRECTLY classified as calibration!"
