import cv2
import numpy as np
import torch
import torch.nn as nn

class ECGQualityClassifier(nn.Module):
    """
    Recommended Research Model: EfficientNet-B3
    Used to classify ECG image quality into Accept/Reject.
    """
    def __init__(self, num_classes=2):
        super(ECGQualityClassifier, self).__init__()
        # In a real environment, load pretrained EfficientNet-B3 from torchvision
        # here we define a placeholder architecture for the research spec.
        self.features = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def assess_image_quality(image, config):
    """
    Evaluates whether the ECG image is blurred, overexposed, or underexposed.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Blur Detection (Laplacian Variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurred = lap_var < config['blur_threshold']

    # 2. Exposure & Illumination Detection
    mean_val = np.mean(gray)
    is_overexposed = mean_val > config['brightness_upper']
    is_underexposed = mean_val < config['brightness_lower']

    issues = []
    if is_blurred:
        issues.append(f"Image is blurry (Laplacian Variance: {lap_var:.2f} < {config['blur_threshold']})")
    if is_overexposed:
        issues.append(f"Image is overexposed (Mean Brightness: {mean_val:.2f})")
    if is_underexposed:
        issues.append(f"Image is underexposed (Mean Brightness: {mean_val:.2f})")

    accept = len(issues) == 0
    confidence = 1.0 - (0.3 * len(issues))
    confidence = max(0.1, min(1.0, confidence))

    return {
        "accept": accept,
        "quality_score": lap_var,
        "detected_issues": issues,
        "confidence_score": confidence
    }
