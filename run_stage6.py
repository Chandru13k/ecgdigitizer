import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ecg_digitizer.modules.stage06_wfdb_export import (
    load_stage5_leads,
    get_calibration_factors,
    convert_stage5_to_physical_signals,
    export_wfdb_record,
    perform_roundtrip_validation,
    LEAD_ORDER,
    MAX_QUANT_ERR_MV,
    REF_FS,
    REF_FMT,
    REF_GAIN,
    REF_BASELINE,
    REF_UNITS
)

def process_single_case(case_dir, output_root="stage6_output"):
    case_name = Path(case_dir).name
    out_dir = Path(output_root) / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Stage 05 reconstructed centerlines (Source of Truth)
    leads_data = load_stage5_leads(case_dir)
    
    # 2. Get calibration metadata
    px_per_sec, px_per_mv, px_mm_x, px_mm_y = get_calibration_factors(case_dir)
    
    # 3. Convert Stage 05 pixels to calibrated physical signals
    t_grid, physical_matrix, calibrated_signals = convert_stage5_to_physical_signals(
        leads_data, px_per_sec, px_per_mv, target_fs=REF_FS
    )
    
    # Record name formatting
    record_name = f"{case_name}_wfdb"
    
    # 4. Write WFDB .dat + .hea files
    hea_path, dat_path, sanitized_rec_name = export_wfdb_record(out_dir, record_name, physical_matrix, fs=REF_FS)
    
    # 5. Mandatory Round-Trip Test using wfdb.rdrecord()
    status, metrics, struct_checks, read_signal = perform_roundtrip_validation(
        hea_path, physical_matrix, sanitized_rec_name
    )
    
    # 6. Save validation_metrics.csv
    df_metrics = pd.DataFrame(metrics)
    csv_path = out_dir / "validation_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    # 7. Save validation_report.json
    report_dict = {
        "case_name": case_name,
        "record_name": record_name,
        "overall_status": status,
        "sampling_frequency_hz": REF_FS,
        "sample_count": len(t_grid),
        "duration_seconds": float(t_grid[-1] - t_grid[0]),
        "calibration": {
            "px_per_mm_x": px_mm_x,
            "px_per_mm_y": px_mm_y,
            "px_per_sec": px_per_sec,
            "px_per_mv": px_per_mv
        },
        "target_wfdb_format": {
            "fmt": REF_FMT,
            "adc_gain": REF_GAIN,
            "baseline": REF_BASELINE,
            "units": REF_UNITS
        },
        "max_quantization_tolerance_mv": MAX_QUANT_ERR_MV,
        "structural_validation": struct_checks,
        "lead_metrics_summary": {
            "max_abs_error_mv_across_leads": float(df_metrics["max_abs_error_mv"].max()),
            "mean_mae_mv_across_leads": float(df_metrics["mae_mv"].mean()),
            "mean_rmse_mv_across_leads": float(df_metrics["rmse_mv"].mean()),
            "min_correlation_across_leads": float(df_metrics["correlation"].min())
        }
    }
    
    json_path = out_dir / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    # 8. Generate visual overlay comparison plot (source_vs_roundtrip.png)
    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()
    
    for idx, lead in enumerate(LEAD_ORDER):
        ax = axes[idx]
        src_v = physical_matrix[:, idx]
        read_v = read_signal[:, idx]
        
        ax.plot(t_grid, src_v, label="Stage 05 Source (mV)", color="blue", linewidth=1.5, alpha=0.8)
        ax.plot(t_grid, read_v, label="WFDB Read-Back (mV)", color="red", linestyle="--", linewidth=1.2, alpha=0.9)
        
        ax.set_title(f"Lead {lead} (Max Err: {df_metrics.loc[df_metrics['lead'] == lead, 'max_abs_error_mv'].values[0]*1000:.2f} µV)")
        ax.set_ylabel("mV")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right")
            
    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plot_path = out_dir / "source_vs_roundtrip.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return report_dict, df_metrics

def main():
    print("=" * 60)
    print("STAGE 06: FAITHFUL WFDB EXPORT & ROUND-TRIP VALIDATION")
    print("=" * 60)
    
    stage5_root = Path("stage5_output")
    if len(sys.argv) > 1:
        target = sys.argv[1]
        target_dir = stage5_root / target
        if target_dir.exists():
            case_dirs = [target_dir]
        else:
            case_dirs = [d for d in stage5_root.iterdir() if d.is_dir() and target in d.name]
    else:
        case_dirs = [d for d in stage5_root.iterdir() if d.is_dir() and not d.name.startswith("model_") and not d.name.startswith("audit_")]
        
    if not case_dirs:
        print("Error: No matching Stage 05 output case directories found in stage5_output/")
        return
        
    case_dirs.sort(key=lambda p: p.name)
    print(f"Found {len(case_dirs)} Stage 05 cases.")
    
    # Requirement: Test single case first before processing all cases
    first_case = case_dirs[0]
    print(f"\n--- TESTING SINGLE CASE FIRST: {first_case.name} ---")
    rep_1, df_1 = process_single_case(first_case)
    print(f"Case {first_case.name} Status: {rep_1['overall_status']}")
    print(f"Max Abs Error: {rep_1['lead_metrics_summary']['max_abs_error_mv_across_leads']*1000:.3f} µV (Limit: {MAX_QUANT_ERR_MV*1000:.1f} µV)")
    print(f"Mean Correlation: {rep_1['lead_metrics_summary']['min_correlation_across_leads']:.6f}")
    
    if rep_1['overall_status'] != "PASS":
        print(f"FATAL: Single case test failed! Stopping execution.")
        return
        
    print("\n--- SINGLE CASE PASSED PERFECTLY! PROCESSING ALL REMAINING CASES ---")
    
    all_reports = [rep_1]
    all_metrics = [df_1]
    
    for case_dir in case_dirs[1:]:
        print(f"Processing case: {case_dir.name}...")
        rep, df_m = process_single_case(case_dir)
        all_reports.append(rep)
        all_metrics.append(df_m)
        
    # Global Summary
    all_pass = all(r['overall_status'] == 'PASS' for r in all_reports)
    print("\n" + "=" * 60)
    print("STAGE 06 OVERALL VALIDATION SUMMARY Across 30 Cases")
    print("=" * 60)
    print(f"Total Cases Processed: {len(all_reports)}")
    print(f"All Cases Passed: {all_pass}")
    
    # Write summary table
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
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("stage6_output/stage6_overall_validation_summary.csv", index=False)
    print(f"Saved global summary to stage6_output/stage6_overall_validation_summary.csv")

if __name__ == "__main__":
    main()
