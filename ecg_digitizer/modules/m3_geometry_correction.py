import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    Recommended Research Model: Spatial Transformer Network (STN)
    Used to automatically rotate/crop text or curves back to canonical format.
    """
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/biases with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.compute_grid(x.size(), theta)
        x = F.grid_sample(x, grid)
        return x

def correct_skew_and_geometry(gray_img, config):
    """
    Detects skew angle using Hough Line Transform and rotates the image to align it.
    """
    h, w = gray_img.shape
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, config['hough_threshold'])
    
    angle = 0
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            deg = np.degrees(theta)
            # Find lines that are close to horizontal (90 degrees in Hough space)
            if 45 < deg < 135:
                angles.append(deg - 90)
        if angles:
            angle = np.median(angles)
            
    if abs(angle) > config['min_skew_angle']:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            gray_img, 
            M, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return corrected, angle
        
    return gray_img, angle
