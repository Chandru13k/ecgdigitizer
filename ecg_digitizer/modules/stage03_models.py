import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class Stage3UNet(nn.Module):
    """
    Lightweight, CPU-friendly U-Net architecture for Stage 3 vertical separator segmentation.
    Uses 3 downsampling stages to maintain high local spatial resolution and low training time.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(Stage3UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(32, 16)
        
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d1 = self.up1(x4)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)
        
        logits = self.outc(d3)
        return logits
