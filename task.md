# Task Checklist: End-to-End Pipeline Execution (Stage 1 to Stage 6)

## Scope: All 133 ECG Cases in `ECG_D/`

- [x] Preparation & Git Isolation
  - [x] Align on 133-case scope with User
  - [x] Switch to / create experiment branch `experiment/e2e-all-133-stages`
- [x] Stage 1: Render high-resolution 300 DPI PNG images for all 133 input PDFs (`run_stage1.py`)
- [x] Stage 2: Detect ECG panel and perform 12-lead ROI grid extraction
  - [x] Stage 2.1: Automatic panel bounding box detection (`run_stage2_1.py`)
  - [x] Stage 2.2: Boundary snapping & panel refinement (`run_stage2_2.py`)
- [x] Stage 3: Targeted grid removal, template matching & glyph disentanglement (`run_stage3.py`)
- [x] Stage 4: Geometric 12-lead ROI partitioning + Rhythm II extraction (`run_stage4.py`)
- [x] Stage 5: Deep waveform semantic segmentation & 1D centerline trace extraction (`run_stage5.py`)
- [x] Stage 5.2: Subpixel centerline extraction & landmark analysis (`run_stage5_2.py`)
- [x] Stage 6: Faithful WFDB export (500 Hz physical signals) & round-trip validation (`run_stage6.py`)
- [x] Quality Audit & Verification:
  - [x] Audit Stage 5 segmentation & centerline fidelity across cases (`audit_stage5.py`)
  - [x] Audit Stage 6 round-trip error & correlation metrics (`pipeline_validation_audit.csv`)
- [x] Visualization & Reporting:
  - [x] Verify visual debug overlays and signal comparison plots (`source_vs_roundtrip.png`, `stage5_debug.png`)
  - [x] Update `ECG_DIGITIZATION_COMPLETE_TECHNICAL_REPORT.md` with 133-case results
  - [x] Git commit & finalize branch merge

