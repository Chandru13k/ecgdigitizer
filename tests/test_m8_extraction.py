import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ecg_digitizer'))
from modules.m8_waveform_extraction import extract_waveform_coordinates

def test_vertical_qrs():
    print("Running Test: test_vertical_qrs")
    
    # Create a dummy image with a massive vertical spike
    h, w = 1000, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Baseline
    mask[900:905, 0:50] = 255
    mask[900:905, 150:200] = 255
    
    # Vertical spike (QRS) spanning from y=100 to y=900
    # Center x=100
    mask[100:905, 95:105] = 255
    
    signal, metrics, vis = extract_waveform_coordinates(mask, {})
    
    valid_sig = signal[~np.isnan(signal)]
    max_amp = np.max(valid_sig) if len(valid_sig) > 0 else 0
    min_amp = np.min(valid_sig) if len(valid_sig) > 0 else 0
    
    # The physical height is 900 - 100 = 800
    # Our signal is inverted (h - y), so baseline at y=900 is amp=100
    # Peak at y=100 is amp=900
    print(f"Extracted peak amplitude: {max_amp}")
    print(f"Extracted baseline amplitude: {min_amp}")
    
    if max_amp >= 895 and min_amp <= 105:
        print("PASS: Vertical QRS perfectly preserved in 1D extraction.")
    else:
        print("FAIL: Vertical QRS clipped.")
        
def test_downward_qrs():
    print("Running Test: test_downward_qrs")
    
    h, w = 1000, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Baseline
    mask[100:105, 0:50] = 255
    mask[100:105, 150:200] = 255
    
    # Downward spike
    mask[100:905, 95:105] = 255
    
    signal, metrics, vis = extract_waveform_coordinates(mask, {})
    
    valid_sig = signal[~np.isnan(signal)]
    max_amp = np.max(valid_sig) if len(valid_sig) > 0 else 0
    min_amp = np.min(valid_sig) if len(valid_sig) > 0 else 0
    
    # Baseline at y=100 is amp=900
    # Trough at y=900 is amp=100
    print(f"Extracted peak amplitude: {max_amp}")
    print(f"Extracted baseline amplitude: {min_amp}")
    
    if min_amp <= 105 and max_amp >= 895:
        print("PASS: Downward QRS perfectly preserved in 1D extraction.")
    else:
        print("FAIL: Downward QRS clipped.")

if __name__ == "__main__":
    test_vertical_qrs()
    test_downward_qrs()
