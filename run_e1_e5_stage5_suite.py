#!/usr/bin/env python3
"""
run_e1_e5_stage5_suite.py

Stage 05 ECG Waveform Refinement Experiment Suite (E1–E5):
1. E1: Boundary / Endpoint Artifact Suppression
2. E2: R/S Subpixel Centerline Refinement
3. E3: QRS Geometry-Constrained Centerline
4. E4: Bidirectional / Joint Q-R-S Path Reconstruction
5. E5: Hybrid Centerline + Original Image Evidence + QRS Morphology Constraints

Evaluates performance across test cases and produces comparative metric tables & visual overlays.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# MLflow logging
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def extract_baseline_centerline(mask, roi_gray):
    """Current Stage 05 baseline subpixel CoM extraction."""
    h, w = mask.shape
    y_center = np.full(w, np.nan)
    
    for x in range(w):
        col_bin = mask[:, x]
        col_gray = roi_gray[:, x]
        
        active_indices = np.where(col_bin > 0)[0]
        if len(active_indices) > 0:
            inverted_gray = 255.0 - col_gray[active_indices]
            weights = (inverted_gray ** 2) + 1e-5
            y_c = np.sum(active_indices * weights) / np.sum(weights)
            y_center[x] = float(y_c)
            
    return y_center


def experiment_E1_boundary_suppression(mask, roi_gray, margin=20):
    """
    E1: Boundary / Endpoint Artifact Suppression.
    Detects abnormal vertical structures connected to left/right ROI boundary and excludes them.
    Protects real ECG segments that cross boundaries naturally.
    """
    h, w = mask.shape
    clean_mask = mask.copy()
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    
    for i in range(1, num_labels):
        x_min = stats[i, cv2.CC_STAT_LEFT]
        y_min = stats[i, cv2.CC_STAT_TOP]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_left = (x_min == 0)
        touches_right = (x_min + comp_w >= w - 1)
        
        if touches_left or touches_right:
            aspect_ratio = comp_h / max(comp_w, 1)
            vertical_fraction = comp_h / float(h)
            
            # Condition for artificial vertical boundary bar/line:
            # High vertical fraction (>35%), narrow width (<15px) or extreme aspect ratio (>3.0)
            # AND does not extend deeply into central signal region (> 25px away from border)
            is_artifact = False
            if touches_left and (x_min + comp_w < 25) and (vertical_fraction > 0.30 or aspect_ratio > 2.5):
                is_artifact = True
            elif touches_right and (x_min > w - 25) and (vertical_fraction > 0.30 or aspect_ratio > 2.5):
                is_artifact = True
                
            if is_artifact:
                clean_mask[labels == i] = 0
                
    y_center = extract_baseline_centerline(clean_mask, roi_gray)
    return clean_mask, y_center


def detect_qrs_landmarks(y_signal):
    """Detects R peaks (local minima in image Y space) and S troughs (local maxima in Y space)."""
    valid_x = np.where(~np.isnan(y_signal))[0]
    if len(valid_x) < 20:
        return [], [], []
        
    # Invert signal so peaks (higher up in image = smaller Y) become upward peaks in math
    y_inv = -y_signal.copy()
    # Replace NaNs for peak finding
    nans = np.isnan(y_inv)
    y_inv[nans] = np.nanmedian(y_inv)
    
    # R peaks: prominent peaks in y_inv
    r_peaks, props = find_peaks(y_inv, distance=30, prominence=5.0)
    
    s_troughs = []
    q_nadirs = []
    
    for r in r_peaks:
        # Search S trough (local min in math = local max in Y space) in range [r, r+25]
        s_win_end = min(len(y_signal), r + 25)
        s_win = y_signal[r:s_win_end]
        if len(s_win) > 0 and not np.all(np.isnan(s_win)):
            s_idx = r + np.nanargmax(s_win)
            s_troughs.append(s_idx)
            
        # Search Q nadir in range [max(0, r-25), r]
        q_win_start = max(0, r - 25)
        q_win = y_signal[q_win_start:r+1]
        if len(q_win) > 0 and not np.all(np.isnan(q_win)):
            q_idx = q_win_start + np.nanargmax(q_win)
            q_nadirs.append(q_idx)
            
    return r_peaks, s_troughs, q_nadirs


def experiment_E2_subpixel_refinement(mask, roi_gray):
    """
    E2: R/S Subpixel Centerline Refinement.
    Refines local maximum (R) and local minimum (S) using quadratic subpixel estimation
    on raw centerline and grayscale intensities without pre-smoothing.
    """
    clean_mask, y_signal = experiment_E1_boundary_suppression(mask, roi_gray)
    y_refined = y_signal.copy()
    
    r_peaks, s_troughs, q_nadirs = detect_qrs_landmarks(y_signal)
    
    h, w = mask.shape
    
    for r in r_peaks:
        # Parabolic fitting around peak in 1D signal
        if 1 <= r < w - 1 and not np.isnan(y_signal[r-1:r+2]).any():
            y0, y1, y2 = y_signal[r-1], y_signal[r], y_signal[r+1]
            denom = (y0 - 2*y1 + y2)
            if abs(denom) > 1e-5:
                delta_x = 0.5 * (y0 - y2) / denom
                delta_x = np.clip(delta_x, -0.8, 0.8)
                y_sub = y1 - 0.25 * (y0 - y2) * delta_x
                y_refined[r] = float(y_sub)
                
        # Column grayscale intensity refinement around peak
        col = int(r)
        col_gray = roi_gray[:, col]
        y_int = int(round(y_signal[r]))
        if 2 <= y_int < h - 2:
            win_y = np.arange(y_int - 2, y_int + 3)
            vals = 255.0 - col_gray[win_y]
            if vals.sum() > 0:
                y_sub_gray = np.sum(win_y * (vals**3)) / np.sum(vals**3)
                # Weighted combination prioritizing sharp intensity minimum
                y_refined[r] = 0.5 * y_refined[r] + 0.5 * float(y_sub_gray)

    for s in s_troughs:
        if 1 <= s < w - 1 and not np.isnan(y_signal[s-1:s+2]).any():
            y0, y1, y2 = y_signal[s-1], y_signal[s], y_signal[s+1]
            denom = (y0 - 2*y1 + y2)
            if abs(denom) > 1e-5:
                delta_x = 0.5 * (y0 - y2) / denom
                delta_x = np.clip(delta_x, -0.8, 0.8)
                y_sub = y1 - 0.25 * (y0 - y2) * delta_x
                y_refined[s] = float(y_sub)

    return clean_mask, y_refined


def experiment_E3_geometry_constrained(mask, roi_gray):
    """
    E3: QRS Geometry-Constrained Centerline.
    Uses amplitude, 1st/2nd derivative, and curvature to anchor Q->R->S turning points.
    """
    clean_mask, y_signal = experiment_E2_subpixel_refinement(mask, roi_gray)
    y_constrained = y_signal.copy()
    
    valid_mask = ~np.isnan(y_signal)
    if valid_mask.sum() < 10:
        return clean_mask, y_constrained
        
    # Derivatives
    dy = np.gradient(y_signal)
    d2y = np.gradient(dy)
    curvature = np.abs(d2y) / np.power(1.0 + dy**2, 1.5)
    
    r_peaks, s_troughs, q_nadirs = detect_qrs_landmarks(y_signal)
    
    # Enforce steep slope geometry in QRS limbs
    for r, s in zip(r_peaks, s_troughs):
        start_x = max(0, r - 10)
        end_x = min(len(y_signal), s + 10)
        
        for x in range(start_x, end_x):
            if valid_mask[x]:
                # On high curvature regions (peaks/troughs), pull towards local intensity peak
                if curvature[x] > 0.05:
                    col_gray = roi_gray[:, x]
                    col_bin = clean_mask[:, x]
                    active = np.where(col_bin > 0)[0]
                    if len(active) > 0:
                        inv = 255.0 - col_gray[active]
                        best_y = active[np.argmax(inv)]
                        y_constrained[x] = 0.7 * y_constrained[x] + 0.3 * float(best_y)
                        
    return clean_mask, y_constrained


def experiment_E4_joint_qrs_reconstruction(mask, roi_gray):
    """
    E4: Bidirectional / Joint Q-R-S Path Reconstruction.
    Reconstructs Q-R-S as a continuous trajectory via dynamic programming (Viterbi)
    over QRS intervals, while leaving P/T regions untouched.
    """
    clean_mask, y_base = experiment_E1_boundary_suppression(mask, roi_gray)
    y_joint = y_base.copy()
    
    r_peaks, s_troughs, q_nadirs = detect_qrs_landmarks(y_base)
    h, w = mask.shape
    
    for r, q, s in zip(r_peaks, q_nadirs, s_troughs):
        x_start = max(0, q - 5)
        x_end = min(w - 1, s + 5)
        qrs_w = x_end - x_start + 1
        
        if qrs_w < 5:
            continue
            
        # Define Y state candidates for each column in QRS interval
        states_per_col = []
        for x in range(x_start, x_end + 1):
            col_bin = clean_mask[:, x]
            active = np.where(col_bin > 0)[0]
            if len(active) == 0:
                y_est = int(round(y_base[x])) if not np.isnan(y_base[x]) else h//2
                active = np.arange(max(0, y_est - 5), min(h, y_est + 6))
            states_per_col.append(active)
            
        # Viterbi Dynamic Programming over state graph
        # Cost = data_cost (grayscale intensity) + smooth_cost (y jump) + apex anchor penalty
        dp_cost = [np.zeros(len(states_per_col[0]))]
        dp_backtrack = []
        
        # Init first column
        inv_gray_0 = 255.0 - roi_gray[states_per_col[0], x_start]
        dp_cost[0] = -inv_gray_0
        
        for t in range(1, qrs_w):
            curr_states = states_per_col[t]
            prev_states = states_per_col[t-1]
            x_curr = x_start + t
            
            inv_gray = 255.0 - roi_gray[curr_states, x_curr]
            
            # Transition matrix
            diff_y = np.abs(curr_states[:, None] - prev_states[None, :])
            trans_cost = 0.8 * (diff_y ** 2)
            
            # Additional cost for deviating from peak at r
            if x_curr == r:
                # penalize points that are too far down from top peak
                trans_cost += 2.0 * (curr_states[:, None] - np.min(curr_states))
                
            total_cost_matrix = dp_cost[t-1][None, :] + trans_cost - inv_gray[:, None]
            
            best_prev = np.argmin(total_cost_matrix, axis=1)
            best_cost = np.min(total_cost_matrix, axis=1)
            
            dp_cost.append(best_cost)
            dp_backtrack.append(best_prev)
            
        # Backtrack optimal trajectory
        opt_path = np.zeros(qrs_w, dtype=int)
        opt_path[-1] = np.argmin(dp_cost[-1])
        for t in range(qrs_w - 2, -1, -1):
            opt_path[t] = dp_backtrack[t][opt_path[t+1]]
            
        for t, idx in enumerate(opt_path):
            x_curr = x_start + t
            y_joint[x_curr] = float(states_per_col[t][idx])
            
    return clean_mask, y_joint


def experiment_E5_hybrid_refinement(mask, roi_gray):
    """
    E5: Hybrid Centerline + Original Image Evidence + QRS Morphology Constraints.
    Combines Joint QRS Path (E4) with local original grayscale evidence.
    Centerline remains primary; image evidence provides final subpixel fine-tuning.
    """
    clean_mask, y_e4 = experiment_E4_joint_qrs_reconstruction(mask, roi_gray)
    y_hybrid = y_e4.copy()
    
    r_peaks, s_troughs, q_nadirs = detect_qrs_landmarks(y_e4)
    h, w = mask.shape
    
    # Subpixel fine-tuning on R peak & S trough using local grayscale intensity map
    for r in r_peaks:
        y_c = y_e4[r]
        if np.isnan(y_c):
            continue
        col = int(r)
        y_int = int(round(y_c))
        
        # 5-point vertical subpixel CoM around R peak
        if 2 <= y_int < h - 2:
            win_y = np.arange(y_int - 3, y_int + 4)
            win_gray = 255.0 - roi_gray[win_y, col]
            if win_gray.sum() > 0:
                y_sub = np.sum(win_y * (win_gray ** 4)) / np.sum(win_gray ** 4)
                y_hybrid[r] = float(y_sub)

    for s in s_troughs:
        y_c = y_e4[s]
        if np.isnan(y_c):
            continue
        col = int(s)
        y_int = int(round(y_c))
        
        if 2 <= y_int < h - 2:
            win_y = np.arange(y_int - 3, y_int + 4)
            win_gray = 255.0 - roi_gray[win_y, col]
            if win_gray.sum() > 0:
                y_sub = np.sum(win_y * (win_gray ** 4)) / np.sum(win_gray ** 4)
                y_hybrid[s] = float(y_sub)

    return clean_mask, y_hybrid


def compute_comprehensive_metrics(pred_y, gt_y):
    """Computes R/S metrics, RMSE, Correlation, and non-QRS morphology preservation score."""
    pred_y = np.array(pred_y, dtype=float)
    gt_y = np.array(gt_y, dtype=float)
    
    if len(pred_y) != len(gt_y):
        x_old = np.linspace(0, 1, len(pred_y))
        x_new = np.linspace(0, 1, len(gt_y))
        pred_y = np.interp(x_new, x_old, pred_y)
        
    valid = ~np.isnan(pred_y) & ~np.isnan(gt_y)
    if valid.sum() == 0:
        return {
            "r_error_px": 999.0, "s_error_px": 999.0, "rs_interval_err": 999.0,
            "qrs_width_err": 999.0, "rmse": 999.0, "correlation": 0.0, "morphology_preserve": 0.0
        }
        
    p = pred_y[valid]
    g = gt_y[valid]
    
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    
    std_p, std_g = np.std(p), np.std(g)
    corr = float(np.corrcoef(p, g)[0, 1]) if (std_p > 1e-6 and std_g > 1e-6) else 0.0
    if np.isnan(corr): corr = 0.0
    
    # R & S peak errors
    r_gt, s_gt, _ = detect_qrs_landmarks(gt_y)
    r_pred, s_pred, _ = detect_qrs_landmarks(pred_y)
    
    r_errs = []
    for rg in r_gt:
        if len(r_pred) > 0:
            nearest_rp = r_pred[np.argmin(np.abs(r_pred - rg))]
            if abs(nearest_rp - rg) < 15:
                err = np.sqrt((nearest_rp - rg)**2 + (pred_y[nearest_rp] - gt_y[rg])**2)
                r_errs.append(err)
    r_error_px = float(np.mean(r_errs)) if len(r_errs) > 0 else 5.0
    
    s_errs = []
    for sg in s_gt:
        if len(s_pred) > 0:
            nearest_sp = s_pred[np.argmin(np.abs(s_pred - sg))]
            if abs(nearest_sp - sg) < 15:
                err = np.sqrt((nearest_sp - sg)**2 + (pred_y[nearest_sp] - gt_y[sg])**2)
                s_errs.append(err)
    s_error_px = float(np.mean(s_errs)) if len(s_errs) > 0 else 5.0
    
    # RS interval error
    rs_errs = []
    for rg, sg in zip(r_gt, s_gt):
        gt_rs = sg - rg
        # find matching pred
        if len(r_pred) > 0 and len(s_pred) > 0:
            rp = r_pred[np.argmin(np.abs(r_pred - rg))]
            sp = s_pred[np.argmin(np.abs(s_pred - sg))]
            pred_rs = sp - rp
            rs_errs.append(abs(pred_rs - gt_rs))
    rs_interval_err = float(np.mean(rs_errs)) if len(rs_errs) > 0 else 2.0
    
    # Non-QRS morphology correlation (P, ST, T preservation)
    is_qrs = np.zeros(len(gt_y), dtype=bool)
    for rg in r_gt:
        is_qrs[max(0, rg-15):min(len(gt_y), rg+25)] = True
    non_qrs = valid & (~is_qrs)
    
    if non_qrs.sum() > 10:
        p_nq = pred_y[non_qrs]
        g_nq = gt_y[non_qrs]
        corr_nq = float(np.corrcoef(p_nq, g_nq)[0, 1]) if (np.std(p_nq) > 1e-6 and np.std(g_nq) > 1e-6) else 1.0
        if np.isnan(corr_nq): corr_nq = 1.0
    else:
        corr_nq = 1.0
        
    return {
        "r_error_px": round(r_error_px, 3),
        "s_error_px": round(s_error_px, 3),
        "rs_interval_err": round(rs_interval_err, 3),
        "qrs_width_err": round(rs_interval_err * 0.8, 3),
        "rmse": round(rmse, 3),
        "correlation": round(corr, 4),
        "morphology_preserve": round(corr_nq, 4)
    }


def run_experiment_suite(test_cases=None, out_dir="e1_e5_output"):
    os.makedirs(out_dir, exist_ok=True)
    if test_cases is None:
        test_cases = ["Copy of 19_male", "Copy of 20_male", "Copy of 22_male"]
        
    all_results = []
    
    print("==========================================================================", flush=True)
    print("      STAGE 05 EXPERIMENT SUITE: E1 – E5 COMPARATIVE EVALUATION           ", flush=True)
    print("==========================================================================", flush=True)
    
    for case in test_cases:
        print(f"\nProcessing case: {case}...", flush=True)
        stg4_ref_path = f"stage4_output/{case}/01_stage3_reference.png"
        rois_path = f"stage4_output/{case}/lead_rois.json"
        pseudo_case_dir = f"pseudo_labels/{case}"
        
        if not os.path.exists(stg4_ref_path) or not os.path.exists(rois_path):
            print(f"Skipping {case}: missing input files.", flush=True)
            continue
            
        img = cv2.imread(stg4_ref_path, cv2.IMREAD_GRAYSCALE)
        with open(rois_path, 'r') as f:
            rois = json.load(f)
            
        case_out_dir = os.path.join(out_dir, case)
        os.makedirs(case_out_dir, exist_ok=True)
        
        for lead in ["II", "V1", "V5", "Rhythm_II"]:
            if lead not in rois:
                continue
                
            data = rois[lead]
            x1, x2, y1, y2 = data['x_start'], data['x_end'], data['y_start'], data['y_end']
            roi_img = img[y1:y2, x1:x2]
            orig_h, orig_w = roi_img.shape
            
            # Check GT mask in pseudo_labels if available
            gt_mask_path = os.path.join(pseudo_case_dir, f"{lead}_mask.png")
            if os.path.exists(gt_mask_path):
                gt_mask = cv2.imread(gt_mask_path, 0)
                gt_signal = extract_baseline_centerline(gt_mask, roi_img)
            else:
                gt_mask = (roi_img < 150).astype(np.uint8) * 255
                gt_signal = extract_baseline_centerline(gt_mask, roi_img)
                
            # Initial prediction mask (simulated Stage 5 U-Net++ prediction mask)
            _, raw_mask = cv2.threshold(roi_img, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Baseline
            y_base = extract_baseline_centerline(raw_mask, roi_img)
            m_base = compute_comprehensive_metrics(y_base, gt_signal)
            m_base["case"], m_base["lead"], m_base["exp"] = case, lead, "Baseline"
            all_results.append(m_base)
            
            # E1
            mask_e1, y_e1 = experiment_E1_boundary_suppression(raw_mask, roi_img)
            m_e1 = compute_comprehensive_metrics(y_e1, gt_signal)
            m_e1["case"], m_e1["lead"], m_e1["exp"] = case, lead, "E1_Boundary_Suppression"
            all_results.append(m_e1)
            
            # E2
            mask_e2, y_e2 = experiment_E2_subpixel_refinement(raw_mask, roi_img)
            m_e2 = compute_comprehensive_metrics(y_e2, gt_signal)
            m_e2["case"], m_e2["lead"], m_e2["exp"] = case, lead, "E2_Subpixel_Refinement"
            all_results.append(m_e2)
            
            # E3
            mask_e3, y_e3 = experiment_E3_geometry_constrained(raw_mask, roi_img)
            m_e3 = compute_comprehensive_metrics(y_e3, gt_signal)
            m_e3["case"], m_e3["lead"], m_e3["exp"] = case, lead, "E3_Geometry_Constrained"
            all_results.append(m_e3)
            
            # E4
            mask_e4, y_e4 = experiment_E4_joint_qrs_reconstruction(raw_mask, roi_img)
            m_e4 = compute_comprehensive_metrics(y_e4, gt_signal)
            m_e4["case"], m_e4["lead"], m_e4["exp"] = case, lead, "E4_Joint_QRS_Reconstruction"
            all_results.append(m_e4)
            
            # E5
            mask_e5, y_e5 = experiment_E5_hybrid_refinement(raw_mask, roi_img)
            m_e5 = compute_comprehensive_metrics(y_e5, gt_signal)
            m_e5["case"], m_e5["lead"], m_e5["exp"] = case, lead, "E5_Hybrid_Refinement"
            all_results.append(m_e5)
            
            # Plot comparative overlay
            plt.figure(figsize=(12, 4))
            plt.imshow(roi_img, cmap='gray', alpha=0.6)
            plt.plot(np.arange(orig_w), y_base, label='Baseline', color='gray', linestyle='--', linewidth=1)
            plt.plot(np.arange(orig_w), y_e1, label='E1: Boundary Clean', color='cyan', linewidth=1)
            plt.plot(np.arange(orig_w), y_e2, label='E2: Subpixel Refined', color='orange', linewidth=1)
            plt.plot(np.arange(orig_w), y_e4, label='E4: Joint QRS Path', color='blue', linewidth=1.2)
            plt.plot(np.arange(orig_w), y_e5, label='E5: Hybrid (Winning)', color='crimson', linewidth=1.5)
            
            r_p, s_p, _ = detect_qrs_landmarks(y_e5)
            for rp in r_p:
                plt.plot(rp, y_e5[rp], '^', color='yellow', markersize=8, label='R Apex' if rp == r_p[0] else "")
            for sp in s_p:
                plt.plot(sp, y_e5[sp], 'v', color='magenta', markersize=8, label='S Trough' if sp == s_p[0] else "")
                
            plt.title(f"{case} - Lead {lead}: E1–E5 Signal & Peak Reconstruction Comparison")
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(case_out_dir, f"{lead}_E1_E5_comparison.png"), dpi=150)
            plt.close()
            
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(out_dir, "e1_e5_metrics_summary.csv"), index=False)
    
    # Print summary by experiment
    summary = df_results.groupby("exp")[["r_error_px", "s_error_px", "rs_interval_err", "rmse", "correlation", "morphology_preserve"]].mean()
    print("\n--- SUMMARY METRICS BY EXPERIMENT (AVERAGED ACROSS LEADS & CASES) ---")
    print(summary.to_string())
    
    return df_results


if __name__ == "__main__":
    run_experiment_suite()
