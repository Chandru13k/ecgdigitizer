---
name: validation_agent
description: Audits results, validates IoU, Precision, Recall, and coordinates metric compliance checks.
---

# Validation Agent Skill

This skill governs verification and auditing of metrics for the ECG Digitization pipeline.

## 📋 Objective
To ensure zero regression, high pixel retention, zero-morphology displacement, and robust U-Net++ segmentation performance.

## ⚙️ Workflow & Steps
1. **Locate Audits**: Identify appropriate audit scripts based on the stage:
   - Stage 3: `audit_stage3.py`
   - Stage 4: `audit_stage4.py`
   - Stage 5: `audit_stage5.py`
   - Stage 5.2: `audit_stage5_2.py`
2. **Execute Validation**: Trigger the verification suite to check:
   - Forensics: non-red pixel loss ratios
   - Bounding boxes
   - Overlap coefficients
   - Trace subpixel distances
3. **Verify Metrics Thresholds**:
   - IoU > 0.15 (provisional deep learning segmentation limits)
   - Displacement == 0 (Branch A immutability requirement)
4. **Issue Report**: Document findings in `debug_metrics.json` and report any validation failures to the **Master Orchestrator**.
