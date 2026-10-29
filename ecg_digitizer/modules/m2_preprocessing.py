import cv2

def preprocess_image(image, config):
    """
    Applies CLAHE and Bilateral Filtering to enhance the ECG waveform
    while keeping the signal margins sharp.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(
        clipLimit=config['clahe_clip'], 
        tileGridSize=tuple(config['clahe_grid'])
    )
    enhanced = clahe.apply(gray)
    
    # 2. Bilateral Filter (reduces paper grid texture noise while keeping signal edges sharp)
    filtered = cv2.bilateralFilter(
        enhanced, 
        d=config['bilateral_d'], 
        sigmaColor=config['bilateral_color'], 
        sigmaSpace=config['bilateral_space']
    )
    
    return filtered
