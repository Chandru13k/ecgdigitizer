# Task Checklist: Eliminate Stage 05 Starting Vertical Line Artifact

- [x] Refine boundary artifact suppression in `run_stage5.py`
  - [x] Detect & strip merged vertical stem pixels near borders ($x \in [0, 25]$ and $x \in [w-25, w-1]$)
  - [x] Exclude top/bottom crop frame border pixels ($y=0$ or $y=h-1$) at boundaries
  - [x] Leave non-signal initial columns as `NaN` without CoM mid-height spikes
  - [x] Protect natural horizontal ECG traces crossing borders
- [x] Regenerate `stage5_output/` across cases
- [x] Audit output using `audit_stage5.py`
- [x] Verify visual overlays and centerline plots for 0 starting vertical line artifacts (PASSED)
