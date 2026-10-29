import pytest
import cv2
import numpy as np
import os
import json

from ecg_digitizer.modules.stage03_5_glyph_disentanglement import (
    _analyze_overlap_case,
    perform_glyph_disentanglement,
    run_stage3_5_glyph_disentanglement,
    get_char_template
)

def test_analyze_overlap_case():
    h, w = 120, 200
    roi_img = np.zeros((h, w), dtype=np.uint8)
    text_mask_roi = np.zeros((h, w), dtype=np.uint8)
    
    # Test 1: Already removed
    text_mask_roi[50:60, 50:60] = 255
    assert _analyze_overlap_case(roi_img, text_mask_roi) == "already_removed"
    
    # Reset
    text_mask_roi.fill(0)
    
    # Test 2: Not found
    assert _analyze_overlap_case(roi_img, text_mask_roi) == "not_found"
    
    # Test 3: Partially overlapped (small components)
    cv2.rectangle(roi_img, (10, 10), (30, 30), 255, -1)
    assert _analyze_overlap_case(roi_img, text_mask_roi) == "partially overlapped"
    
    # Test 4: Fully fused (large component > 50% width)
    cv2.rectangle(roi_img, (0, 50), (150, 70), 255, -1)
    assert _analyze_overlap_case(roi_img, text_mask_roi) == "fully fused"

def test_perform_glyph_disentanglement():
    h, w = 120, 200
    roi_cand = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a horizontal ECG line (will be protected)
    cv2.line(roi_cand, (0, 60), (200, 60), 255, 3)
    
    # Draw a "V" intersecting the line (size matched with templates)
    cv2.putText(roi_cand, "V", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 255, 2, cv2.LINE_AA)
    
    # Grayscale image (Thresholded: white text on black background)
    roi_ap_gray = np.zeros((h, w), dtype=np.uint8)
    roi_ap_gray[roi_cand > 0] = 255
    
    # Templates
    templates = {"V": get_char_template("V", "fake_dir")}
    
    removal_mask = perform_glyph_disentanglement(roi_cand, roi_ap_gray, ["V"], templates)
    
    # The removal mask should contain some pixels of the "V"
    # But it MUST NOT contain pixels on the horizontal ECG line (y=60)
    assert np.sum(removal_mask) > 0
    
    # ECG line should be fully protected (invariant)
    # Check y=60 across the whole width
    assert np.sum(removal_mask[58:63, :]) == 0

def test_run_stage3_5():
    h, w = 1376, 2916
    stage3_candidate = np.zeros((h, w), dtype=np.uint8)
    stage3_text_mask = np.zeros((h, w), dtype=np.uint8)
    active_panel = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Add a mock separator line so ROIs resolve
    cv2.line(stage3_candidate, (720, 0), (720, h), 255, 1)
    cv2.line(stage3_candidate, (1440, 0), (1440, h), 255, 1)
    cv2.line(stage3_candidate, (2160, 0), (2160, h), 255, 1)
    
    metrics = {
        "separator_x_coordinates": [720, 1440, 2160]
    }
    
    stage3_5_mask, diagnostic_images, out_metrics = run_stage3_5_glyph_disentanglement(
        stage3_candidate, stage3_text_mask, metrics, active_panel
    )
    
    assert "stage3_5_classifications" in out_metrics
    assert "10_stage3_5_text_removal" in diagnostic_images
    assert stage3_5_mask.shape == (h, w)
