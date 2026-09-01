import pytest
import shutil
from pathlib import Path
from ecg_digitizer.modules.stage01_pdf_renderer import process_pdf

BASE_DIR = Path(r"C:\Users\Chandru\OneDrive\Desktop\6th sem\trial")
ECG_DIR = BASE_DIR / "ECG_D"
OUTPUT_DIR = BASE_DIR / "stage1_output"

def test_stage01_pdf_renderer_19_male():
    pdf_path = ECG_DIR / "Copy of 19_male.pdf"
    assert pdf_path.exists(), "Test PDF 19_male not found"
    
    # Run the stage
    metadata, report = process_pdf(pdf_path, OUTPUT_DIR, dpi=300)
    
    assert report["overall_status"] == "PASS", f"Validation failed: {report}"
    assert len(metadata) > 0, "No metadata generated"
    
    for meta in metadata:
        assert Path(meta["output_path"]).exists(), "PNG not found"
        assert Path(meta["preview_path"]).exists(), "Preview PNG not found"
        assert meta["rendered_width"] > 0
        assert meta["rendered_height"] > 0
        assert meta["color_mode"] == "RGB"

def test_stage01_pdf_renderer_20_male():
    pdf_path = ECG_DIR / "Copy of 20_male.pdf"
    assert pdf_path.exists(), "Test PDF 20_male not found"
    
    # Run the stage
    metadata, report = process_pdf(pdf_path, OUTPUT_DIR, dpi=300)
    
    assert report["overall_status"] == "PASS", f"Validation failed: {report}"
    assert len(metadata) > 0, "No metadata generated"
    
    for meta in metadata:
        assert Path(meta["output_path"]).exists(), "PNG not found"
        assert Path(meta["preview_path"]).exists(), "Preview PNG not found"
        assert meta["rendered_width"] > 0
        assert meta["rendered_height"] > 0
        assert meta["color_mode"] == "RGB"

# Optional: Clean up function if needed
