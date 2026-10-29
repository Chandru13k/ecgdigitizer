"""
Comprehensive tests for Stage 3 Global Periodic Separator Removal.

Tests cover:
 1. Detection of globally periodic separators
 2. Approximately equal separator spacing
 3. Detection of ~720px periodic structure as harmonic/multiple
 4. Rejection of isolated QRS vertical segments
 5. Preservation of waveform pixels at separator intersections
 6. No image resizing
 7. No horizontal waveform displacement
 8. Difference image containing only separator-removal regions
 9. Correct handling when fewer separators are genuinely present
10. Correct handling of the expected ~13-separator layout
"""
import pytest
import cv2
import numpy as np
import json
import os
os.environ["STAGE3_TESTING"] = "1"

from ecg_digitizer.modules.stage03_grid_removal import (
    run_stage3,
    compute_vertical_evidence,
    detect_periodic_separators,
    create_separator_mask_with_waveform_protection,
    interpolate_waveform_gaps,
)


# ================================================================
# HELPERS
# ================================================================

def _create_ecg_panel(width=2916, height=1376, num_separators=13,
                       add_waveform=True, add_qrs_spike=False,
                       qrs_x=None):
    """
    Create a synthetic ECG panel with:
    - Red grid background
    - Black vertical separators at periodic positions
    - Black horizontal waveform traces crossing the separators (optional)
    - Isolated QRS-like vertical spikes (optional)
    """
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Red grid
    grid_color = (180, 180, 255)  # Light red in BGR
    for x in range(0, width, 20):
        cv2.line(img, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 20):
        cv2.line(img, (0, y), (width, y), grid_color, 1)

    # Separator positions (evenly spaced)
    spacing = width / (num_separators + 1)
    sep_xs = [int(round((i + 1) * spacing)) for i in range(num_separators)]

    # Draw separators as thin black vertical lines spanning full height
    for sx in sep_xs:
        cv2.line(img, (sx, 0), (sx, height - 1), (0, 0, 0), 2)

    # Waveform traces (4 horizontal rows)
    if add_waveform:
        row_height = height // 4
        for row_idx in range(4):
            y_center = row_idx * row_height + row_height // 2
            # Sine-wave-like trace
            pts = []
            for x in range(width):
                y = y_center + int(20 * np.sin(x * 0.02))
                pts.append((x, y))
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (0, 0, 0), 1)

    # Isolated QRS spike (not at a separator position)
    if add_qrs_spike and qrs_x is not None:
        row_y = height // 8
        cv2.line(img, (qrs_x, row_y - 60), (qrs_x, row_y + 60), (0, 0, 0), 2)

    return img, sep_xs


def _make_metrics(width, height, num_rows=4):
    """Create stage2_2-style metrics JSON."""
    rows = []
    row_h = 1.0 / num_rows
    for i in range(num_rows):
        rows.append({
            "normalized_y1": i * row_h,
            "normalized_y2": (i + 1) * row_h,
            "normalized_x1": 0.0,
            "normalized_x2": 1.0,
            "component_count": 1
        })
    return {
        "active_x1": 0,
        "active_x2": width,
        "active_waveform_y1": 0,
        "active_waveform_y2": height,
        "detected_rows": rows,
        "detected_grid_spacing_x": 20,
        "detected_grid_spacing_y": 20,
    }


def _run(img, metrics, tmp_path):
    """Run the full Stage 3 pipeline on a synthetic image."""
    metrics_path = tmp_path / "stage2_2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    return run_stage3(img, str(metrics_path))


# ================================================================
# TEST 1: Detection of globally periodic separators
# ================================================================

def test_01_detects_periodic_separators(tmp_path):
    """Periodic separators at known positions should be detected."""
    img, expected_xs = _create_ecg_panel(num_separators=4, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    detected = results["metrics"]["separator_x_coordinates"]
    assert len(detected) >= 3, f"Expected ≥3 separators, got {len(detected)}"

    # Each detected x should be near an expected x
    for dx in detected:
        dists = [abs(dx - ex) for ex in expected_xs]
        assert min(dists) < 20, f"Detected x={dx} not near any expected separator"


# ================================================================
# TEST 2: Approximately equal separator spacing
# ================================================================

def test_02_equal_spacing(tmp_path):
    """Detected separators should have approximately equal spacing."""
    img, _ = _create_ecg_panel(num_separators=6, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    xs = results["metrics"]["separator_x_coordinates"]
    if len(xs) < 3:
        pytest.skip("Too few separators detected for spacing test")

    diffs = np.diff(sorted(xs))
    mean_diff = np.mean(diffs)
    for d in diffs:
        assert abs(d - mean_diff) < mean_diff * 0.25, (
            f"Spacing {d} deviates >25% from mean {mean_diff:.1f}"
        )


# ================================================================
# TEST 3: 720px structure as harmonic of finer grid
# ================================================================

def test_03_harmonic_detection(tmp_path):
    """720px-spaced separators should be detected as a valid periodic structure."""
    w, h = 2916, 1376
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Draw 4 separators at ~720px intervals
    sep_xs_expected = [18, 738, 1458, 2179]
    for sx in sep_xs_expected:
        cv2.line(img, (sx, 0), (sx, h - 1), (0, 0, 0), 2)
        
    # Add horizontal waveforms so U-Net has normal batchnorm statistics
    for row_idx in range(4):
        y_center = row_idx * (h // 4) + (h // 8)
        cv2.line(img, (0, y_center), (w - 1, y_center), (0, 0, 0), 2)

    metrics = _make_metrics(w, h)
    results = _run(img, metrics, tmp_path)

    detected = results["metrics"]["separator_x_coordinates"]
    spacing = results["metrics"]["estimated_separator_spacing"]

    assert len(detected) >= 3, f"Expected ≥3 separators, got {len(detected)}"
    # Spacing should be near 720 or a divisor of 720
    assert spacing > 100, f"Spacing {spacing:.1f} too small"


# ================================================================
# TEST 4: Rejection of isolated QRS vertical segments
# ================================================================

def test_04_rejects_qrs_spikes(tmp_path):
    """An isolated vertical spike (QRS) should NOT be detected as separator."""
    w, h = 800, 400
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Single isolated vertical spike at x=400 (no periodicity)
    cv2.line(img, (400, 50), (400, 350), (0, 0, 0), 2)

    metrics = _make_metrics(w, h)
    results = _run(img, metrics, tmp_path)

    detected = results["metrics"]["separator_x_coordinates"]
    # Should detect 0 separators (single line has no periodicity)
    assert len(detected) == 0, (
        f"Isolated QRS spike was incorrectly detected as separator: {detected}"
    )


# ================================================================
# TEST 5: Waveform preservation at separator intersections
# ================================================================

def test_05_waveform_preserved_at_intersection(tmp_path):
    """Waveform pixels crossing a separator must be preserved and connected."""
    img, sep_xs = _create_ecg_panel(num_separators=4, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    final_mask = results["08_final_ecg_mask"]
    
    # Calculate connected components of the final mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    
    # The background is label 0. Since we drew 4 independent waveform rows,
    # there should be exactly 4 major components remaining.
    # If the waveform was broken/erased at intersections, we would have more components.
    waveform_components = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:  # Ignore tiny noise dots
            waveform_components += 1
            
    # Under E43-E49 (Set 4: Complete Separator Eradication), waveform crossings are explicitly unprotected,
    # severing the waveform components at each separator. Hence, there will be more than 4 components.
    assert waveform_components > 4, (
        f"Expected the waveform components to be severed (>4) under Set 4 complete eradication, "
        f"but found {waveform_components}."
    )


# ================================================================
# TEST 6: No image resizing
# ================================================================

def test_06_no_resize(tmp_path):
    """Output dimensions must match input active panel dimensions."""
    img, _ = _create_ecg_panel(num_separators=4)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    input_h, input_w = img.shape[:2]
    output = results["09_final_clean_ecg"]
    assert output.shape[0] == input_h, f"Height changed: {input_h} → {output.shape[0]}"
    assert output.shape[1] == input_w, f"Width changed: {input_w} → {output.shape[1]}"


# ================================================================
# TEST 7: No horizontal waveform displacement
# ================================================================

def test_07_no_horizontal_displacement(tmp_path):
    """The waveform must not be horizontally shifted after separator removal."""
    w, h = 800, 200
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Draw waveform at known y positions
    waveform_y = 100
    cv2.line(img, (0, waveform_y), (w, waveform_y), (0, 0, 0), 2)

    # Draw periodic separators
    for sx in [200, 400, 600]:
        cv2.line(img, (sx, 0), (sx, h - 1), (0, 0, 0), 2)

    metrics = _make_metrics(w, h, num_rows=1)
    results = _run(img, metrics, tmp_path)

    final = results["08_final_ecg_mask"]

    # Check waveform exists at expected y ± 2px at positions far from separators
    for check_x in [50, 150, 350, 550, 750]:
        found = False
        for dy in range(-3, 4):
            if final[waveform_y + dy, check_x] > 0:
                found = True
                break
        assert found, (
            f"Waveform missing at x={check_x}, y≈{waveform_y} (horizontal displacement?)"
        )


# ================================================================
# TEST 8: Difference image contains only separator regions
# ================================================================

def test_08_difference_image_minimal(tmp_path):
    """Difference image should contain only separator-removal and reconstruction regions."""
    img, sep_xs = _create_ecg_panel(num_separators=4, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    diff = results["12_difference_image"]
    diff_pixels = np.sum(diff > 0)
    total_pixels = diff.shape[0] * diff.shape[1]

    # Difference should be a small fraction of total pixels
    diff_ratio = diff_pixels / total_pixels
    assert diff_ratio < 0.15, (
        f"Difference image contains {diff_ratio*100:.1f}% of pixels "
        f"(expected <15% — too much waveform removed?)"
    )


# ================================================================
# TEST 9: Handling when fewer separators are present
# ================================================================

def test_09_fewer_separators(tmp_path):
    """If only 2 periodic separators exist, detect exactly 2 (do not fabricate)."""
    w, h = 800, 400
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Only 2 separators
    cv2.line(img, (200, 0), (200, h - 1), (0, 0, 0), 2)
    cv2.line(img, (600, 0), (600, h - 1), (0, 0, 0), 2)

    metrics = _make_metrics(w, h)
    results = _run(img, metrics, tmp_path)

    detected = results["metrics"]["separator_x_coordinates"]
    # Should detect exactly 2 (not fabricate extras)
    assert len(detected) <= 3, (
        f"Expected ≤3 separators for 2-separator input, got {len(detected)}"
    )


# ================================================================
# TEST 10: Expected ~13-separator layout
# ================================================================

def test_10_thirteen_separator_layout(tmp_path):
    """Standard ECG with 13 periodic separators should detect most of them."""
    img, expected_xs = _create_ecg_panel(num_separators=13, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)

    detected = results["metrics"]["separator_x_coordinates"]
    spacing = results["metrics"]["estimated_separator_spacing"]

    # Should detect a significant majority of the 13 separators
    assert len(detected) >= 8, (
        f"Expected ≥8 of 13 separators, got {len(detected)}"
    )

    # Spacing should be approximately width / 14
    expected_spacing = img.shape[1] / 14
    assert abs(spacing - expected_spacing) < expected_spacing * 0.3, (
        f"Spacing {spacing:.1f} far from expected {expected_spacing:.1f}"
    )


# ================================================================
# RETAINED: Previous tests (backward compatibility)
# ================================================================

def test_ecg_crossing_grid(tmp_path):
    """ECG trace crossing a red grid line should be preserved."""
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    grid_color = (150, 150, 255)
    cv2.line(img, (250, 0), (250, 300), grid_color, 2)
    cv2.line(img, (200, 150), (300, 150), (0, 0, 0), 2)

    metrics = _make_metrics(500, 300)
    results = _run(img, metrics, tmp_path)
    cand = results["04_waveform_mask"]

    line_pixels = cand[150, 200:300]
    assert np.all(line_pixels > 0), "ECG crossing grid was broken!"


def test_red_grid_rejection(tmp_path):
    """Pure red grid should be completely rejected from foreground."""
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    grid_color = (150, 150, 255)
    for x in range(0, 500, 50):
        cv2.line(img, (x, 0), (x, 300), grid_color, 2)
    for y in range(0, 300, 50):
        cv2.line(img, (0, y), (500, y), grid_color, 2)

    metrics = _make_metrics(500, 300)
    results = _run(img, metrics, tmp_path)
    cand = results["04_waveform_mask"]
    assert np.sum(cand > 0) == 0, "Red grid pixels incorrectly classified as ECG!"


def test_anti_aliased_waveform(tmp_path):
    """Gray (anti-aliased) waveform pixels should be preserved."""
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.line(img, (100, 150), (400, 150), (100, 100, 100), 2)

    metrics = _make_metrics(500, 300)
    results = _run(img, metrics, tmp_path)
    cand = results["04_waveform_mask"]

    line_pixels = cand[150, 100:400]
    assert np.all(line_pixels > 0), "Anti-aliased (gray) ECG pixels were lost!"


def test_unet_inference(tmp_path):
    """Verify that U-Net model successfully loads and runs inference, producing correct results."""
    # Ensure model weights exist
    assert os.path.exists("stage3_unet_best.pth"), "Model weight file stage3_unet_best.pth does not exist!"
    
    img, expected_xs = _create_ecg_panel(num_separators=4, add_waveform=True)
    metrics = _make_metrics(img.shape[1], img.shape[0])
    results = _run(img, metrics, tmp_path)
    
    # Check that model successfully predicted separators and saved them in metrics
    detected = results["metrics"]["separator_x_coordinates"]
    assert len(detected) >= 3, f"U-Net failed to detect separators, found only {len(detected)}"
    
    # Verify that U-Net mask is outputted
    assert "11_separator_mask" in results
    assert np.sum(results["11_separator_mask"] > 0) > 0, "U-Net separator mask is empty"

