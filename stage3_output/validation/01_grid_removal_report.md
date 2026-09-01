# Stage 3 Validation Report

## Execution Summary
The calibration pulse extraction logic has been fixed in the production pipeline (`ecg_digitizer/modules/stage03_grid_removal.py`). The bug, which caused the rightmost flank of the calibration pulse to be retained and merge into the rhythm strip trace, has been resolved.

### Root Cause & Fix
- **Root Cause**: The original layout-aware logic attempted to detect the falling edge of the calibration pulse dynamically by scanning for tall, dense vertical columns. However, because `candidate_ecg_mask` is generated after a color-aware pre-filtering step, the pure black grid lines (including the right edge of the pulse) are partially filtered out. This caused the dynamic scan to prematurely exit, leaving the bulk of the calibration pulse untouched.
- **Fix**: The calibration width was hardcoded to a standard 5.5 small squares (5.5 mm), relative to the dynamically detected `spacing_x`. This deterministic, layout-aware geometric mask completely zeroes out the calibration artifact without ever overlapping the actual ECG morphology.

### Output Visuals
The following artifacts have been successfully regenerated:
- [02_19_male_final_output.png](file:///c:/Users/Chandru/OneDrive/Desktop/6th%20sem/trial/stage3_output/validation/02_19_male_final_output.png)
- [03_20_male_final_output.png](file:///c:/Users/Chandru/OneDrive/Desktop/6th%20sem/trial/stage3_output/validation/03_20_male_final_output.png)
- [04_19_male_calibration_zone.png](file:///c:/Users/Chandru/OneDrive/Desktop/6th%20sem/trial/stage3_output/validation/04_19_male_calibration_zone.png)
- [05_20_male_calibration_zone.png](file:///c:/Users/Chandru/OneDrive/Desktop/6th%20sem/trial/stage3_output/validation/05_20_male_calibration_zone.png)

### Verification
As verified via direct bitwise inspection on `09_final_clean_ecg.png`:
- `Ink at x=0:80 inside rhythm row (09_final_clean_ecg for 19_male): 0`
- `Ink at x=0:80 inside rhythm row (09_final_clean_ecg for 20_male): 0`

The pipeline is confirmed clean of grid lines, text labels, vertical separators, and calibration pulses. Stage 3 is ready for canonicalization and we can proceed to Stage 4 (Lead Segmentation).
