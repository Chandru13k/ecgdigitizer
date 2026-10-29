---
name: research_manager
description: Writes documentation, chronicles experiments, and builds the ECG_DIGITIZATION_COMPLETE_TECHNICAL_REPORT.md.
---

# Research Manager Skill

This skill governs logging and writing detailed reports for the ECG Digitization project.

## 📋 Objective
To maintain complete clarity and trace of the system by updating the technical report and documenting limitations, dataset leakage, and evaluation results.

## ⚙️ Workflow & Steps
1. **Document Audit**: Read the existing `ECG_DIGITIZATION_COMPLETE_TECHNICAL_REPORT.md` to identify outdated sections or newly completed stages.
2. **Result Compilation**: Gather metrics from the **Validation Agent** and plots from the **Visualization Manager**.
3. **Draft Updates**: Use GitHub Markdown syntax to update the reports.
4. **Leakage & Limitations Audit**: Update the "Limitations" and "Dataset Documentation" sections whenever data, labels, or models change.
5. **Verify Formatting**: Ensure all internal images, videos, and file paths are correctly structured with absolute file URIs.
