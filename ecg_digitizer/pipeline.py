import yaml
import os
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path

# Import all modules
from ecg_digitizer.modules.m1_quality_assessment import assess_image_quality
from ecg_digitizer.modules.m2_preprocessing import preprocess_image
from ecg_digitizer.modules.m3_geometry_correction import correct_skew_and_geometry
from ecg_digitizer.modules.m4_layout_detection import detect_ecg_layout
from ecg_digitizer.modules.m5_lead_segmentation import segment_all_leads
from ecg_digitizer.modules.m6_grid_removal import remove_grid
from ecg_digitizer.modules.m7_waveform_segmentation import segment_waveform
from ecg_digitizer.modules.m8_waveform_extraction import extract_waveform_coordinates
from ecg_digitizer.modules.m9_reconstruction import reconstruct_continuous_signal
from ecg_digitizer.modules.m10_calibration import calibrate_signal
from ecg_digitizer.modules.m11_enhancement import enhance_signal
from ecg_digitizer.modules.m12_validation import validate_reconstruction
from ecg_digitizer.modules.m13_export import export_digitized_ecg

class ECGDigitizationPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_input(self, input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.pdf':
            print(f"Loading PDF natively: {input_path}")
            try:
                images = convert_from_path(input_path, dpi=self.config['system'].get('dpi', 300))
                if not images:
                    raise ValueError("Failed to render PDF pages.")
                # We assume single page ECG for this test case
                img = np.array(images[0])
                # Convert RGB to BGR for OpenCV
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                import fitz
                print("pdf2image failed, falling back to PyMuPDF (fitz)")
                doc = fitz.open(input_path)
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=self.config['system'].get('dpi', 300))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
        else:
            print(f"Loading Image: {input_path}")
            return cv2.imread(input_path)

    def run_digitization(self, input_path, base_name):
        print(f"\n==================================================")
        print(f"Starting Digitization Pipeline for: {base_name}")
        print(f"==================================================")
        
        # Create debug directory
        out_dir = Path(self.config['system']['output_dir'])
        debug_dir = out_dir / base_name / self.config['system'].get('debug_dir', 'debug')
        debug_dir.mkdir(parents=True, exist_ok=True)

        color_image = self.load_input(input_path)
        if color_image is None:
            print("FAILED to load input file.")
            return {"status": "FAIL", "reason": "Could not load input image/PDF"}

        cv2.imwrite(str(debug_dir / "01_original.png"), color_image)

        # M1: Quality Assessment
        qa = assess_image_quality(color_image, self.config['m1_quality'])
        if not qa['accept']:
            print(f" -> QA Decision: REJECT. Warning: {qa['detected_issues']}")
            return {"status": "FAIL", "reason": "Failed Image Quality Check"}

        # M2 & M3: Preprocessing & Geometry
        preprocessed = preprocess_image(color_image, self.config['m2_preprocessing'])
        aligned_gray, skew_angle = correct_skew_and_geometry(preprocessed, self.config['m3_geometry'])
        
        # Warp color image by same matrix
        aligned_color = color_image.copy()
        if abs(skew_angle) > self.config['m3_geometry']['min_skew_angle']:
            h, w = aligned_gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            aligned_color = cv2.warpAffine(color_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(debug_dir / "02_corrected.png"), aligned_color)

        # M4: Layout Detection
        layout_bboxes, layout_vis = detect_ecg_layout(aligned_color, self.config['m4_layout'])
        cv2.imwrite(str(debug_dir / "03_layout_detection.png"), layout_vis)

        # M6 & M7: Grid Removal & Segment Waveform
        # M6: Grid Removal
        waveform_dict = remove_grid(aligned_color, self.config['m6_grid_removal'])
        
        # M7: Waveform Segmentation
        waveform_dict = segment_waveform(waveform_dict, self.config['m7_waveform_segmentation'])
        
        binary_waveform = waveform_dict["waveform_final"]

        # Debug Exports for M6/M7
        cv2.imwrite(str(debug_dir / "05_grid_candidate.png"), waveform_dict["grid_candidate"])
        cv2.imwrite(str(debug_dir / "06_waveform_initial.png"), waveform_dict["waveform_initial"])
        cv2.imwrite(str(debug_dir / "07_waveform_protected.png"), waveform_dict["waveform_protected"])
        cv2.imwrite(str(debug_dir / "08_waveform_final.png"), waveform_dict["waveform_final"])

        # M5: Lead Segmentation
        binary_patches, raw_color_patches, roi_diagnostics = segment_all_leads(binary_waveform, aligned_color, layout_bboxes, self.config['m4_layout'])

        # Draw ROI diagnostics
        roi_vis = aligned_color.copy()
        for name, diag in roi_diagnostics.items():
            iy1, ix1, iy2, ix2 = diag['initial']
            fy1, fx1, fy2, fx2 = diag['final']
            cv2.rectangle(roi_vis, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2) # Blue = initial
            cv2.rectangle(roi_vis, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2) # Green = final
            cv2.putText(roi_vis, name, (fx1 + 5, fy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "04_roi_diagnostics.png"), roi_vis)

        # Process each lead
        calibrated_leads_raw = {}
        calibrated_leads_clean = {}
        validation_reports = {}
        time_axis = None
        
        trace_overlay_img = aligned_color.copy()

        for name, patch in binary_patches.items():
            if np.sum(patch) == 0:
                print(f" -> Warning: Lead {name} is empty. Skipping.")
                continue

            # M8: Waveform Extraction (DP trace)
            pixel_signal, extraction_metrics, skeleton_vis = extract_waveform_coordinates(patch, self.config['m8_extraction'])
            cv2.imwrite(str(debug_dir / f"08_skeleton_{name}.png"), skeleton_vis)

            # Draw trace over the overlay image
            f_ymin, f_xmin, f_ymax, f_xmax = roi_diagnostics[name]['final']
            for dx, y_val in enumerate(pixel_signal):
                if not np.isnan(y_val):
                    px, py = int(f_xmin + dx), int(f_ymin + y_val)
                    if 0 <= py < trace_overlay_img.shape[0] and 0 <= px < trace_overlay_img.shape[1]:
                        trace_overlay_img[py, px] = (0, 255, 0) # Green trace

            # M9: Signal Reconstruction
            reconstructed_raw = reconstruct_continuous_signal(pixel_signal, self.config)

            # M10: Calibration
            times, volts_mv_raw = calibrate_signal(reconstructed_raw, self.config['m10_calibration'])
            calibrated_leads_raw[name] = volts_mv_raw
            if time_axis is None or len(times) > len(time_axis):
                time_axis = times

            # M11: Signal Enhancement
            volts_mv_clean = enhance_signal(volts_mv_raw, self.config['m11_enhancement'])
            calibrated_leads_clean[name] = volts_mv_clean

            # M12: Validation
            validation_reports[name] = validate_reconstruction(
                volts_mv_raw, 
                volts_mv_clean,
                times,
                extraction_metrics,
                self.config['m12_validation']
            )
            
        cv2.imwrite(str(debug_dir / "09_trace_overlays.png"), trace_overlay_img)

        # Assess overall pass/warn/fail
        overall_status = "PASS"
        for name, report in validation_reports.items():
            if report['status'] == 'FAIL':
                overall_status = "FAIL"
                break
            elif report['status'] == 'WARNING' and overall_status != "FAIL":
                overall_status = "WARNING"

        # M13: Export
        export_digitized_ecg(
            str(out_dir / base_name),
            base_name,
            time_axis,
            calibrated_leads_raw,
            calibrated_leads_clean,
            validation_reports,
            overall_status,
            self.config
        )
        
        print(f"Digitization Complete. Status: {overall_status}")
        return {
            "status": overall_status,
            "time": time_axis,
            "leads_clean": calibrated_leads_clean,
            "reports": validation_reports
        }
