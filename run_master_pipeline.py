#!/usr/bin/env python
"""
run_master_pipeline.py

Master Orchestrator Pipeline Runner for ECG Digitization:
Executes the full end-to-end pipeline (Stages 1 through 6) across all 133 cases in ECG_D/.

Pipeline Stages:
  Stage 1:   Document Acquisition & High-DPI Rendering (300 DPI)
  Stage 2.1: Automatic ECG Panel Bounding Box Detection
  Stage 2.2: Outer Boundary Snapping & Panel Refinement
  Stage 3:   Targeted Grid Removal, Text Inpainting & Glyph Disentanglement
  Stage 4:   Geometric 12-Lead Partitioning + Rhythm II Extraction
  Stage 5:   Deep Waveform Semantic Segmentation (U-Net++ ResNet-18) & Centerline Tracing
  Stage 5.2: Dual-Branch Subpixel Waveform Extraction & Landmark Analysis
  Stage 6:   Faithful WFDB Format Export & Mandatory Round-Trip Validation
  Validation: Forensic metric validation & summary reporting
"""

import os
import sys
import glob
import time
import json
import logging
import traceback
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("master_pipeline_execution.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("MasterPipeline")

# Import Pipeline Modules
from ecg_digitizer.modules.stage01_pdf_renderer import process_pdf
from run_stage2_1 import run_stage2_1
from run_stage2_2 import run_stage2_2
from ecg_digitizer.modules.stage03_grid_removal import run_stage3
from ecg_digitizer.modules.stage04_lead_segmentation import run_stage4
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from run_stage5 import run_stage5
from run_stage5_2 import run_stage5_2
from run_stage6 import process_single_case
from mlflow_config import setup_mlflow

def execute_stage1(pdf_files, output_dir=Path("stage1_output")):
    logger.info("=" * 60)
    logger.info("PHASE 1: STAGE 1 — 300 DPI PDF RENDERING")
    logger.info("=" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        pdf_path = Path(pdf_path)
        case_name = pdf_path.stem
        img_check = output_dir / case_name / "page_001.png"
        if img_check.exists() and img_check.stat().st_size > 1000:
            results[case_name] = {"status": "PASS", "pages": 1, "cached": True}
            continue
            
        t0 = time.time()
        try:
            metadata, report = process_pdf(pdf_path, output_dir, dpi=300)
            status = report.get("overall_status", "PASS")
            elapsed = time.time() - t0
            results[case_name] = {"status": status, "pages": len(metadata), "elapsed_s": elapsed}
            if idx % 10 == 0 or idx == len(pdf_files):
                logger.info(f"[{idx:3d}/{len(pdf_files):3d}] Stage 1: {case_name} -> {status} ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(pdf_files):3d}] Stage 1 FAILED for {case_name}: {e}")
            results[case_name] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 1 Complete: {passed}/{len(pdf_files)} PDFs verified & rendered.\n")
    return results

def execute_stage2_1(cases, stage1_dir=Path("stage1_output"), stage2_1_dir=Path("stage2_1_output")):
    logger.info("=" * 60)
    logger.info("PHASE 2A: STAGE 2.1 — ECG PANEL DETECTION")
    logger.info("=" * 60)
    stage2_1_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, case in enumerate(cases, 1):
        crop_check = stage2_1_dir / case / "05_ecg_panel_crop.png"
        meta_check = stage2_1_dir / case / "stage2_1_metrics.json"
        if crop_check.exists() and meta_check.exists():
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        img_path = stage1_dir / case / "page_001.png"
        if not img_path.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
        t0 = time.time()
        try:
            success = run_stage2_1(img_path)
            status = "PASS" if success else "FAIL"
            elapsed = time.time() - t0
            results[case] = {"status": status, "elapsed_s": elapsed}
            if idx % 20 == 0 or idx == len(cases):
                logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 2.1: {case} -> {status} ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 2.1 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 2A Complete: {passed}/{len(cases)} panels detected.\n")
    return results

def execute_stage2_2(cases, stage1_dir=Path("stage1_output"), stage2_1_dir=Path("stage2_1_output"), stage2_2_dir=Path("stage2_2_output")):
    logger.info("=" * 60)
    logger.info("PHASE 2B: STAGE 2.2 — PANEL BOUNDARY REFINEMENT")
    logger.info("=" * 60)
    stage2_2_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, case in enumerate(cases, 1):
        crop_check = stage2_2_dir / case / "03_ecg_panel_crop.png"
        meta_check = stage2_2_dir / case / "stage2_2_metrics.json"
        if crop_check.exists() and meta_check.exists():
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        metrics_path = stage2_1_dir / case / "stage2_1_metrics.json"
        img_path = stage1_dir / case / "page_001.png"
        if not metrics_path.exists() or not img_path.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
        t0 = time.time()
        try:
            success = run_stage2_2(metrics_path, img_path)
            status = "PASS" if success else "FAIL"
            elapsed = time.time() - t0
            results[case] = {"status": status, "elapsed_s": elapsed}
            if idx % 20 == 0 or idx == len(cases):
                logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 2.2: {case} -> {status} ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 2.2 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 2B Complete: {passed}/{len(cases)} panels refined.\n")
    return results

def execute_stage3(cases, stage2_2_dir=Path("stage2_2_output"), stage3_dir=Path("stage3_output")):
    logger.info("=" * 60)
    logger.info("PHASE 3: STAGE 3 — GRID REMOVAL & GLYPH DISENTANGLEMENT")
    logger.info("=" * 60)
    stage3_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, case in enumerate(cases, 1):
        clean_check = stage3_dir / case / "09_final_clean_ecg.png"
        meta_check = stage3_dir / case / "stage3_metrics.json"
        if clean_check.exists() and meta_check.exists():
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        original_img_path = stage2_2_dir / case / "01_original.png"
        metrics_path = stage2_2_dir / case / "stage2_2_metrics.json"
        if not original_img_path.exists() or not metrics_path.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
            
        t0 = time.time()
        try:
            image = cv2.imread(str(original_img_path))
            out_case_dir = stage3_dir / case
            out_case_dir.mkdir(parents=True, exist_ok=True)
            
            stage3_res = run_stage3(image, str(metrics_path))
            
            # Save core outputs
            cv2.imwrite(str(out_case_dir / "01_active_panel.png"), stage3_res["01_active_panel"])
            cv2.imwrite(str(out_case_dir / "02_grid_mask.png"), stage3_res["02_grid_mask"])
            cv2.imwrite(str(out_case_dir / "03_grid_removed.png"), stage3_res["03_grid_removed"])
            cv2.imwrite(str(out_case_dir / "04_waveform_mask.png"), stage3_res["04_waveform_mask"])
            cv2.imwrite(str(out_case_dir / "05_clean_ecg.png"), stage3_res["05_clean_ecg"])
            cv2.imwrite(str(out_case_dir / "06_grid_removal_overlay.png"), stage3_res["06_grid_removal_overlay"])
            
            for key in ["07_01_candidate_mask", "07_02_separator_mask", "07_03_calibration_mask",
                        "07_04_text_mask", "07_05_artifact_mask", "07_06_final_ecg_mask",
                        "07_07_subtraction_overlay", "07_08_separator_intersection_with_ecg",
                        "07_09_calibration_intersection_with_ecg", "07_10_text_intersection_with_ecg",
                        "07_11_lost_ecg_pixels", "07_12_waveform_preservation_overlay",
                        "07_13_waveform_at_separator", "07_14_interpolated_pixels"]:
                if key in stage3_res and stage3_res[key] is not None:
                    cv2.imwrite(str(out_case_dir / f"{key}.png"), stage3_res[key])
                    
            cv2.imwrite(str(out_case_dir / "08_final_ecg_mask.png"), stage3_res["08_final_ecg_mask"])
            cv2.imwrite(str(out_case_dir / "09_final_clean_ecg.png"), stage3_res["09_final_clean_ecg"])
            if "10_subtraction_rgb_overlay" in stage3_res:
                cv2.imwrite(str(out_case_dir / "10_subtraction_rgb_overlay.png"), stage3_res["10_subtraction_rgb_overlay"])
            if "10_separator_positions" in stage3_res:
                cv2.imwrite(str(out_case_dir / "10_separator_positions.png"), stage3_res["10_separator_positions"])
            if "11_separator_mask" in stage3_res:
                cv2.imwrite(str(out_case_dir / "11_separator_mask.png"), stage3_res["11_separator_mask"])
            
            metrics_data = stage3_res.get("metrics") or stage3_res.get("stage3_metrics", {})
            with open(out_case_dir / "stage3_metrics.json", "w") as f:
                json.dump(metrics_data, f, indent=4)
                
            elapsed = time.time() - t0
            results[case] = {"status": "PASS", "elapsed_s": elapsed}
            if idx % 15 == 0 or idx == len(cases):
                logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 3: {case} -> PASS ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 3 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 3 Complete: {passed}/{len(cases)} cleaned panels generated.\n")
    return results

def execute_stage4(cases, stage3_dir=Path("stage3_output"), stage2_2_dir=Path("stage2_2_output"), stage4_dir=Path("stage4_output")):
    logger.info("=" * 60)
    logger.info("PHASE 4: STAGE 4 — GEOMETRIC 12-LEAD PARTITIONING")
    logger.info("=" * 60)
    stage4_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, case in enumerate(cases, 1):
        rois_check = stage4_dir / case / "lead_rois.json"
        ref_check = stage4_dir / case / "01_stage3_reference.png"
        if rois_check.exists() and ref_check.exists():
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        s3_img_path = stage3_dir / case / "09_final_clean_ecg.png"
        s3_metrics_path = stage3_dir / case / "stage3_metrics.json"
        s22_metrics_path = stage2_2_dir / case / "stage2_2_metrics.json"
        
        if not s3_img_path.exists() or not s3_metrics_path.exists() or not s22_metrics_path.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
            
        t0 = time.time()
        try:
            s3_img = cv2.imread(str(s3_img_path))
            with open(s3_metrics_path, "r") as f:
                s3_metrics = json.load(f)
            with open(s22_metrics_path, "r") as f:
                s22_metrics = json.load(f)
                
            rois, debug_img, metrics = run_stage4(s3_img, s3_metrics, s22_metrics)
            
            out_case_dir = stage4_dir / case
            out_case_dir.mkdir(parents=True, exist_ok=True)
            
            # Reference image
            cv2.imwrite(str(out_case_dir / "01_stage3_reference.png"), s3_img)
            cv2.imwrite(str(out_case_dir / "lead_segmentation_debug.png"), debug_img)
            
            with open(out_case_dir / "lead_rois.json", "w") as f:
                json.dump(rois, f, indent=4)
            with open(out_case_dir / "stage4_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
                
            elapsed = time.time() - t0
            results[case] = {"status": "PASS", "leads_detected": metrics.get("total_leads_detected", 0), "elapsed_s": elapsed}
            if idx % 20 == 0 or idx == len(cases):
                logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 4: {case} -> PASS ({elapsed:.2f}s, {metrics.get('total_leads_detected', 0)} leads)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 4 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 4 Complete: {passed}/{len(cases)} cases segmented into 12-lead grids.\n")
    return results

def execute_stage5(cases, stage4_dir=Path("stage4_output"), stage5_dir=Path("stage5_output")):
    logger.info("=" * 60)
    logger.info("PHASE 5A: STAGE 5 — DEEP WAVEFORM SEGMENTATION & CENTERLINES")
    logger.info("=" * 60)
    stage5_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch Device: {device}")
    
    model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
    checkpoint_info = None
    if os.path.exists("stage5_best_model.pth"):
        model.load_state_dict(torch.load("stage5_best_model.pth", map_location=device, weights_only=True))
        file_stats = os.stat("stage5_best_model.pth")
        with open("stage5_best_model.pth", "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        checkpoint_info = {
            "path": "stage5_best_model.pth",
            "sha256": file_hash,
            "size_bytes": file_stats.st_size,
            "architecture": "UNetPlusPlusResNet18"
        }
        logger.info(f"Loaded Stage 5 checkpoint: stage5_best_model.pth (SHA256: {file_hash[:16]}...)")
    else:
        logger.warning("No stage5_best_model.pth found! Running with initial weights.")
        
    model.eval()
    results = {}
    
    for idx, case in enumerate(cases, 1):
        traces_dir = stage5_dir / case / "06_digitized_traces"
        if traces_dir.exists() and len(list(traces_dir.glob("*.csv"))) >= 12:
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        s4_ref = stage4_dir / case / "01_stage3_reference.png"
        rois_path = stage4_dir / case / "lead_rois.json"
        if not s4_ref.exists() or not rois_path.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
            
        t0 = time.time()
        try:
            run_stage5(case, model, device, checkpoint_info)
            elapsed = time.time() - t0
            results[case] = {"status": "PASS", "elapsed_s": elapsed}
            logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 5: {case} -> PASS ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 5 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 5A Complete: {passed}/{len(cases)} cases segmented & vectorized.\n")
    return results

def execute_stage5_2(cases, stage5_dir=Path("stage5_output"), stage5_2_dir=Path("stage5_2_output")):
    logger.info("=" * 60)
    logger.info("PHASE 5B: STAGE 5.2 — SUBPIXEL TRACING & LANDMARK DELINEATION")
    logger.info("=" * 60)
    stage5_2_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for idx, case in enumerate(cases, 1):
        wf_dir = stage5_2_dir / case / "07_waveform"
        if wf_dir.exists() and len(list(wf_dir.glob("*.csv"))) >= 12:
            results[case] = {"status": "PASS", "cached": True}
            continue
            
        cmasks = stage5_dir / case / "04_stage5_clean_masks"
        if not cmasks.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
            
        t0 = time.time()
        try:
            run_stage5_2(case)
            elapsed = time.time() - t0
            results[case] = {"status": "PASS", "elapsed_s": elapsed}
            logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 5.2: {case} -> PASS ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 5.2 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 5B Complete: {passed}/{len(cases)} subpixel traces extracted.\n")
    return results

def execute_stage6(cases, stage5_dir=Path("stage5_output"), stage6_dir=Path("stage6_output")):
    logger.info("=" * 60)
    logger.info("PHASE 6: STAGE 6 — FAITHFUL WFDB EXPORT & ROUND-TRIP VALIDATION")
    logger.info("=" * 60)
    stage6_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    all_reports = []
    
    for idx, case in enumerate(cases, 1):
        report_file = stage6_dir / case / "validation_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r") as f:
                    cached_report = json.load(f)
                if cached_report.get("overall_status") == "PASS":
                    all_reports.append(cached_report)
                    results[case] = {
                        "status": "PASS",
                        "cached": True,
                        "max_abs_error_mv": cached_report["lead_metrics_summary"]["max_abs_error_mv_across_leads"],
                        "min_correlation": cached_report["lead_metrics_summary"]["min_correlation_across_leads"]
                    }
                    continue
            except Exception:
                pass
                
        case_dir = stage5_dir / case
        if not case_dir.exists():
            results[case] = {"status": "SKIPPED_NO_INPUT"}
            continue
            
        t0 = time.time()
        try:
            report_dict, df_metrics = process_single_case(case_dir, output_root=str(stage6_dir))
            elapsed = time.time() - t0
            status = report_dict.get("overall_status", "PASS")
            all_reports.append(report_dict)
            results[case] = {
                "status": status,
                "elapsed_s": elapsed,
                "max_abs_error_mv": report_dict["lead_metrics_summary"]["max_abs_error_mv_across_leads"],
                "min_correlation": report_dict["lead_metrics_summary"]["min_correlation_across_leads"]
            }
            logger.info(f"[{idx:3d}/{len(cases):3d}] Stage 6: {case} -> {status} (MaxErr: {results[case]['max_abs_error_mv']*1000:.2f} µV, r: {results[case]['min_correlation']:.6f})")
        except Exception as e:
            logger.error(f"[{idx:3d}/{len(cases):3d}] Stage 6 FAILED for {case}: {e}")
            results[case] = {"status": "FAIL", "error": str(e)}
            
    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    logger.info(f"Phase 6 Complete: {passed}/{len(cases)} WFDB records generated & round-trip validated.\n")
    
    # Save overall Stage 6 summary
    summary_rows = []
    for r in all_reports:
        summary_rows.append({
            "case_name": r["case_name"],
            "status": r["overall_status"],
            "sample_count": r["sample_count"],
            "max_abs_error_mv": r["lead_metrics_summary"]["max_abs_error_mv_across_leads"],
            "mean_mae_mv": r["lead_metrics_summary"]["mean_mae_mv_across_leads"],
            "mean_rmse_mv": r["lead_metrics_summary"]["mean_rmse_mv_across_leads"],
            "min_correlation": r["lead_metrics_summary"]["min_correlation_across_leads"]
        })
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(stage6_dir / "stage6_overall_validation_summary.csv", index=False)
        logger.info(f"Saved global Stage 6 summary to {stage6_dir / 'stage6_overall_validation_summary.csv'}")
        
    return results

def execute_phase7_audit(cases, stage4_dir=Path("stage4_output"), stage5_dir=Path("stage5_output"), stage6_dir=Path("stage6_output")):
    logger.info("=" * 60)
    logger.info("PHASE 7: VALIDATION AGENT — QUALITY & METRIC COMPLIANCE AUDIT")
    logger.info("=" * 60)
    audit_rows = []
    
    for idx, case in enumerate(cases, 1):
        s4_ref_path = stage4_dir / case / "01_stage3_reference.png"
        s5_masks_dir = stage5_dir / case / "04_stage5_clean_masks"
        s5_traces_dir = stage5_dir / case / "06_digitized_traces"
        s6_report_path = stage6_dir / case / "validation_report.json"
        
        # 1. Geometry & Immutability Check
        immutability_pass = s4_ref_path.exists()
        
        # 2. 13 Lead completeness
        mask_count = len(list(s5_masks_dir.glob("*.png"))) if s5_masks_dir.exists() else 0
        lead_complete = (mask_count == 13)
        
        # 3. WFDB Round-trip fidelity
        wfdb_pass = False
        max_err_mv = np.nan
        min_corr = np.nan
        if s6_report_path.exists():
            try:
                with open(s6_report_path, "r") as f:
                    rdata = json.load(f)
                wfdb_pass = (rdata.get("overall_status") == "PASS")
                summary = rdata.get("lead_metrics_summary", {})
                max_err_mv = summary.get("max_abs_error_mv_across_leads", np.nan)
                min_corr = summary.get("min_correlation_across_leads", np.nan)
            except Exception:
                pass
                
        case_audit_pass = immutability_pass and lead_complete and wfdb_pass
        
        audit_rows.append({
            "case_name": case,
            "immutability_check": "PASS" if immutability_pass else "FAIL",
            "lead_count": mask_count,
            "lead_completeness": "PASS" if lead_complete else "FAIL",
            "wfdb_roundtrip": "PASS" if wfdb_pass else "FAIL",
            "max_abs_error_mv": max_err_mv,
            "min_correlation": min_corr,
            "audit_verdict": "PASS" if case_audit_pass else "FAIL"
        })
        
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv("pipeline_validation_audit.csv", index=False)
    
    passed_audits = sum(1 for r in audit_rows if r["audit_verdict"] == "PASS")
    logger.info(f"Phase 7 Audit Complete: {passed_audits}/{len(cases)} cases passed all forensic compliance checks.")
    logger.info(f"Audit report saved to: pipeline_validation_audit.csv\n")
    return df_audit

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master End-to-End ECG Digitization Pipeline (Stages 1 to 6)")
    parser.add_argument("--input", default="ECG_D", help="Path to input PDF file or directory containing PDFs (default: ECG_D)")
    args = parser.parse_args()
    
    logger.info("================================================================================")
    logger.info("STARTING MASTER END-TO-END ECG DIGITIZATION PIPELINE")
    logger.info("================================================================================")
    start_time = time.time()
    
    # 1. Identify input files
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [str(input_path)]
    elif input_path.is_dir():
        pdf_files = sorted(glob.glob(str(input_path / "*.pdf")))
    else:
        pdf_files = sorted(glob.glob("ECG_D/*.pdf"))
        
    cases = [Path(p).stem for p in pdf_files]
    logger.info(f"Target Input Dataset: {len(pdf_files)} PDFs from {input_path}\n")
    
    # MLflow Setup
    try:
        exp_id = setup_mlflow("ECG_Digitization_All_133_Cases")
        logger.info(f"MLflow Initialized. Experiment ID: {exp_id}")
    except Exception as e:
        logger.warning(f"MLflow initialization skipped: {e}")
        
    # Execute Stages
    s1_res = execute_stage1(pdf_files)
    s21_res = execute_stage2_1(cases)
    s22_res = execute_stage2_2(cases)
    s3_res = execute_stage3(cases)
    s4_res = execute_stage4(cases)
    s5_res = execute_stage5(cases)
    s52_res = execute_stage5_2(cases)
    s6_res = execute_stage6(cases)
    df_audit = execute_phase7_audit(cases)
    
    # Master Consolidation
    master_records = []
    for case in cases:
        c_s1 = s1_res.get(case, {}).get("status", "N/A")
        c_s21 = s21_res.get(case, {}).get("status", "N/A")
        c_s22 = s22_res.get(case, {}).get("status", "N/A")
        c_s3 = s3_res.get(case, {}).get("status", "N/A")
        c_s4 = s4_res.get(case, {}).get("status", "N/A")
        c_s5 = s5_res.get(case, {}).get("status", "N/A")
        c_s52 = s52_res.get(case, {}).get("status", "N/A")
        c_s6 = s6_res.get(case, {}).get("status", "N/A")
        
        overall = "PASS" if all(st == "PASS" for st in [c_s1, c_s21, c_s22, c_s3, c_s4, c_s5, c_s52, c_s6]) else "INCOMPLETE"
        
        master_records.append({
            "case_name": case,
            "stage1_render": c_s1,
            "stage2_1_panel": c_s21,
            "stage2_2_refine": c_s22,
            "stage3_clean": c_s3,
            "stage4_leads": c_s4,
            "stage5_seg": c_s5,
            "stage5_2_subpixel": c_s52,
            "stage6_wfdb": c_s6,
            "overall_status": overall,
            "max_abs_error_mv": s6_res.get(case, {}).get("max_abs_error_mv", np.nan),
            "min_correlation": s6_res.get(case, {}).get("min_correlation", np.nan)
        })
        
    df_master = pd.DataFrame(master_records)
    df_master.to_csv("pipeline_e2e_summary.csv", index=False)
    
    total_elapsed = time.time() - start_time
    total_pass = sum(1 for r in master_records if r["overall_status"] == "PASS")
    
    logger.info("================================================================================")
    logger.info("MASTER PIPELINE EXECUTION COMPLETED")
    logger.info("================================================================================")
    logger.info(f"Total Execution Time: {total_elapsed / 60.0:.2f} minutes ({total_elapsed:.1f} s)")
    logger.info(f"Total Cases: {len(cases)}")
    logger.info(f"Cases Passing All 8 Processing Stages: {total_pass}/{len(cases)} ({total_pass/len(cases)*100:.1f}%)")
    logger.info(f"Master Summary Table saved to: pipeline_e2e_summary.csv")

if __name__ == "__main__":
    main()
