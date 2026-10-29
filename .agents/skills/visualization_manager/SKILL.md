---
name: visualization_manager
description: Renders plots, overlays, bounding box comparisons, and visual debug representations.
---

# Visualization Manager Skill

This skill governs visual inspection asset generation for the ECG Digitization pipeline.

## 📋 Objective
To create intuitive visual evidence that helps researchers audit the pipeline and identify geometric or semantic segmentation failures.

## ⚙️ Workflow & Steps
1. **Identify Stage Inputs/Outputs**: Identify visual targets, such as:
   - Stage 1: Rendered PDF images
   - Stage 2: Boundary comparison overlays
   - Stage 3: Clean ECG grids
   - Stage 4: Geometric lead boundaries
   - Stage 5: Semantic segmentation contour overlays
   - Stage 5.2: Subpixel extraction traces
2. **Generate Plots**: Utilize matplotlib or OpenCV to generate debug images.
3. **Format Images**: Save generated visual assets directly into case directories.
4. **Publish Evidence**: Hand off the file paths of generated images to the **Research Manager** for embedding in the technical report.
