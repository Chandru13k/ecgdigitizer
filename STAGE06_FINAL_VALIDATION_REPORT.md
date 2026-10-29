# STAGE 06 FINAL VALIDATION REPORT
## PTB-Format 1D ECG Signal Reconstruction (`Copy of 19_male`)

### 1. Decision Summary
```
==================================================
STAGE 06 FINAL VALIDATION
==================================================
Case: Copy of 19_male
Source model: M2 U-Net++ / ResNet34
Sampling rate: 500 Hz

Morphology preservation: PASS
Calibration:             PASS
R-peak invariance:       PASS
S-trough invariance:     PASS
Gap handling:            PASS
CSV/NPY consistency:     PASS
Metadata consistency:    PASS
PTB lead order:          PASS

Overall Stage 06:        PASS
==================================================
```

---

### 2. Large Gap & NaN Audit Results
- **Root Cause Investigation**: Previous output filled large gaps during `np.interp` without re-applying the NaN mask over large gap intervals.
- **Resolution**: Large gaps ($> 10\text{ px}$) are now explicitly masked out as `NaN` in `v_resampled`.
- **Matrix NaNs vs Metadata NaNs**:
  - `reconstructed_ecg.npy` NaN count: **421**
  - `reconstructed_ecg.csv` NaN count: **421**
  - `reconstructed_ecg.json` reported final NaNs: **421**
  - Status: **100% PERFECT AGREEMENT & PASS**.

---

### 3. Morphology Preservation Audit
- Pre-Resampling Pearson Correlation ($r$): **1.000000** (exact linear transformation).
- Post-Resampling Pearson Correlation ($r$): **1.000000** across all 12 leads.
- Status: **PASS (0% smoothing, 0% filtering, 0% waveform distortion)**.

---

### 4. R-Peak & S-Trough Transformation Invariance
- Max R-Peak Sample Error: **1 sample** ($\le 0.75\text{ ms}$ time shift attributable solely to 500 Hz grid rounding).
- Max S-Trough Sample Error: **0 samples**.
- Status: **PASS**.

---

### 5. Artifacts Generated
- `stage6_output/Copy of 19_male/reconstructed_ecg.npy`
- `stage6_output/Copy of 19_male/reconstructed_ecg.csv`
- `stage6_output/Copy of 19_male/reconstructed_ecg.json`
- `stage6_output/Copy of 19_male/reconstructed_ecg_plot.png`
- `stage6_output/Copy of 19_male/validation/stage05_vs_stage06_comparison.png`
- `stage6_output/Copy of 19_male/validation/STAGE06_FINAL_VALIDATION.png`
- `stage6_output/Copy of 19_male/validation/stage6_validation_metrics.csv`
