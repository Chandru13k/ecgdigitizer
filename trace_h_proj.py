import cv2
import numpy as np
from pathlib import Path

def trace_h_proj(pdf_name):
    print(f"=== {pdf_name} ===")
    img_path = Path("stage1_output") / pdf_name / "page_001.png"
    image = cv2.imread(str(img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    _, binary_grad_y = cv2.threshold(abs_grad_y, 20, 255, cv2.THRESH_BINARY)
    
    h, w = image.shape[:2]
    k_horiz = max(15, int(w * 0.01))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_horiz, 1))
    lines_h = cv2.morphologyEx(binary_grad_y, cv2.MORPH_OPEN, kernel_h)
    
    h_proj = np.sum(lines_h, axis=1) / 255.0
    max_h = np.max(h_proj)
    print(f"Max h_proj: {max_h}")
    
    count = 0
    for y in range(h-1, 1500, -1):
        if h_proj[y] > max_h * 0.05:
            print(f"y={y}: h_proj={h_proj[y]:.1f}")
            count += 1
            if count > 30:
                break

trace_h_proj("Copy of 20_male")
trace_h_proj("Copy of 19_male")
