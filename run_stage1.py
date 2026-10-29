import argparse
import sys
from pathlib import Path
from ecg_digitizer.modules.stage01_pdf_renderer import process_pdf
import yaml

def main():
    parser = argparse.ArgumentParser(description="Run Stage 1: PDF to High-Resolution Image")
    parser.add_argument("--input", required=True, help="Path to the input PDF file")
    
    args = parser.parse_args()
    pdf_path = Path(args.input)
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
        
    # Read config for DPI, default 300
    base_dir = Path(__file__).parent
    config_path = base_dir / "ecg_digitizer" / "config.yaml"
    dpi = 300
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config and "stage1" in config and "dpi" in config["stage1"]:
                dpi = config["stage1"]["dpi"]
                
    output_dir = base_dir / "stage1_output"
    
    print("=" * 40)
    print("STAGE 1 — PDF RENDERING")
    print("=" * 40)
    print(f"Input: {pdf_path}")
    print(f"DPI: {dpi}")
    
    metadata, report = process_pdf(pdf_path, output_dir, dpi=dpi)
    
    print(f"Pages: {len(metadata)}")
    print("-" * 40)
    
    for i, (meta, page_val) in enumerate(zip(metadata, report["page_validations"])):
        print(f"Page {meta['page_number']}:")
        print(f"PDF size: {meta['pdf_width']} x {meta['pdf_height']} pt")
        print(f"Image size: {meta['rendered_width']} x {meta['rendered_height']} px")
        print(f"Aspect ratio error: {page_val['aspect_ratio_error']:.6f}")
        print(f"Color mode: {meta['color_mode']}")
        
        if page_val["status"] != "PASS":
            print(f"Reasons: {', '.join(page_val['reasons'])}")
        print()
        
    print(f"Result: {report['overall_status']}")
    print("=" * 40)
    print(f"Output Location: {output_dir / pdf_path.stem}")

if __name__ == "__main__":
    main()
