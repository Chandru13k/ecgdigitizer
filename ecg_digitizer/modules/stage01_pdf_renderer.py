import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path

def render_pdf(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(str(pdf_path))
    pages_metadata = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # PDF original dimensions (points, 1/72 inch)
        pdf_rect = page.rect
        pdf_w, pdf_h = pdf_rect.width, pdf_rect.height
        
        # Render
        pix = page.get_pixmap(dpi=dpi)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR if RGB, or add channels if grayscale
        color_mode = "RGB" if pix.n == 3 else ("RGBA" if pix.n == 4 else "GRAYSCALE")
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            color_mode = "RGB" # after conversion
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        render_w, render_h = pix.width, pix.height
        
        # Save exact PNG
        page_str = f"page_{page_num+1:03d}"
        png_path = output_dir / f"{page_str}.png"
        cv2.imwrite(str(png_path), img)
        
        # Save preview (exact same but named _preview.png)
        preview_path = output_dir / f"{page_str}_preview.png"
        cv2.imwrite(str(preview_path), img)
        
        metadata = {
            "source_pdf": str(pdf_path),
            "page_number": page_num + 1,
            "pdf_width": float(pdf_w),
            "pdf_height": float(pdf_h),
            "rendered_width": int(render_w),
            "rendered_height": int(render_h),
            "dpi": dpi,
            "color_mode": color_mode,
            "output_path": str(png_path),
            "preview_path": str(preview_path)
        }
        pages_metadata.append(metadata)
        
    doc.close()
    
    # Save metadata.json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(pages_metadata, f, indent=4)
        
    return pages_metadata

def validate_stage1(metadata_list):
    if not metadata_list:
        return {"status": "FAIL", "reason": "No pages rendered."}
        
    validation_results = []
    all_passed = True
    
    for meta in metadata_list:
        reasons = []
        # Check files exist
        if not Path(meta["output_path"]).exists():
            reasons.append("PNG file not found.")
            
        # Check dimensions
        if meta["rendered_width"] <= 0 or meta["rendered_height"] <= 0:
            reasons.append("Invalid rendered dimensions (<= 0).")
            
        # Check aspect ratio
        pdf_ar = meta["pdf_width"] / meta["pdf_height"] if meta["pdf_height"] > 0 else 0
        img_ar = meta["rendered_width"] / meta["rendered_height"] if meta["rendered_height"] > 0 else 0
        ar_error = abs(pdf_ar - img_ar)
        
        if ar_error > 0.05: # 5% tolerance
            reasons.append(f"Aspect ratio mismatch: PDF={pdf_ar:.4f}, Image={img_ar:.4f}, Error={ar_error:.4f}")
            
        # Check color mode
        if meta["color_mode"] != "RGB":
            reasons.append(f"Color mode is not RGB: {meta['color_mode']}")
            
        page_pass = len(reasons) == 0
        if not page_pass:
            all_passed = False
            
        validation_results.append({
            "page_number": meta["page_number"],
            "status": "PASS" if page_pass else "FAIL",
            "aspect_ratio_error": ar_error,
            "reasons": reasons
        })
        
    report = {
        "overall_status": "PASS" if all_passed else "FAIL",
        "page_validations": validation_results
    }
    
    # Save report
    output_dir = Path(metadata_list[0]["output_path"]).parent
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    return report

def process_pdf(pdf_path, base_output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_dir = Path(base_output_dir) / pdf_name
    
    metadata = render_pdf(pdf_path, output_dir, dpi=dpi)
    report = validate_stage1(metadata)
    
    return metadata, report
