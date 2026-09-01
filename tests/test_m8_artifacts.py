import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m8_waveform_extraction import extract_waveform_coordinates

def test_1_thin_trace():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # thin horizontal line at y=50
    patch[50, :] = 255
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # expected physiological y=50 (inverted it's 100-50 = 50)
    assert np.allclose(signal, 50, atol=1.0)

def test_2_thick_trace():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # thick horizontal line y=48 to 52 (center=50)
    patch[48:53, :] = 255
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # Expected centerline
    assert np.allclose(signal, 50, atol=1.0)

def test_3_thick_qrs_sharp_r():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # baseline y=70 (inverted = 30)
    patch[68:73, 0:30] = 255
    patch[68:73, 71:100] = 255
    
    # sharp R peak at x=50, going up to y=20 (inverted = 80)
    # draw a thick triangle
    for x in range(30, 71):
        if x <= 50:
            y = int(70 - (x - 30) * 2.5)
        else:
            y = int(20 + (x - 50) * 2.5)
        
        y = max(20, min(70, y))
        patch[y-2:y+3, x] = 255
        
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # R peak x-pos should remain around 50
    peak_x = np.argmax(signal)
    assert 48 <= peak_x <= 52
    
    # Amplitude should be well preserved (~80)
    assert signal[peak_x] >= 75

def test_4_thick_qrs_deep_s():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # baseline y=30 (inverted = 70)
    patch[28:33, 0:30] = 255
    patch[28:33, 71:100] = 255
    
    # deep S wave at x=50, going down to y=80 (inverted = 20)
    for x in range(30, 71):
        if x <= 50:
            y = int(30 + (x - 30) * 2.5)
        else:
            y = int(80 - (x - 50) * 2.5)
            
        y = max(30, min(80, y))
        patch[y-2:y+3, x] = 255
        
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    valley_x = np.argmin(signal)
    assert 48 <= valley_x <= 52
    assert signal[valley_x] <= 25

def test_5_vertical_separator():
    patch = np.zeros((100, 100), dtype=np.uint8)
    patch[50, :] = 255  # thin baseline
    
    # vertical separator at x=50
    patch[:, 50] = 255
    
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # should reject the separator and stay near 50
    assert np.allclose(signal, 50, atol=2.0)

def test_6_fragmented_pixels():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # broken baseline
    for x in range(100):
        if x % 5 != 0:
            patch[50, x] = 255
            
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # should interpolate smoothly
    assert np.allclose(signal, 50, atol=1.0)

def test_7_adjacent_artifact():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # baseline y=50
    patch[50, :] = 255
    
    # adjacent artifact at y=20, x=40:60
    # Artifact shouldn't be fully connected over the same x if it's separate, but let's test it
    patch[20:25, 40:60] = 255
    
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # signal should not jump to the artifact (y=20, inverted=80)
    # because of center of mass heavily weighting the true baseline if thicker or artifact is just noise
    # Note: wait, if baseline is 1px and artifact is 5px thick, center of mass shifts. 
    # But thickness penalty should penalize the extreme envelope jump. 
    # Actually, center of mass WILL shift if multiple disconnected blobs exist in same column.
    # The requirement is that artifact doesn't become part of the waveform.
    # We expect signal not to reach 80.
    assert np.max(signal) < 70

def test_8_sharp_qrs_next_to_artifact():
    patch = np.zeros((100, 100), dtype=np.uint8)
    # baseline y=70
    patch[68:73, 0:30] = 255
    patch[68:73, 51:100] = 255
    
    # sharp R at x=40, y=20
    for x in range(30, 51):
        if x <= 40:
            y = int(70 - (x - 30) * 5)
        else:
            y = int(20 + (x - 40) * 5)
        y = max(20, min(70, y))
        patch[y-2:y+3, x] = 255
        
    # artifact at x=42, y=5 to 90 (vertical line)
    patch[5:90, 42] = 255
    
    signal, _, _ = extract_waveform_coordinates(patch, {})
    
    # Check peak is near x=40
    peak_x = np.argmax(signal)
    assert 38 <= peak_x <= 41
    # Check peak amplitude is preserved (~80)
    assert signal[peak_x] >= 75
    # Check artifact didn't cause massive spike
    assert signal[42] < 90
