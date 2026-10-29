import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlusResNet18(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetPlusPlusResNet18, self).__init__()
        
        # Encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify first conv to accept in_channels instead of 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # out: 64, H/4
        self.encoder1 = resnet.layer1 # out: 64, H/4
        self.encoder2 = resnet.layer2 # out: 128, H/8
        self.encoder3 = resnet.layer3 # out: 256, H/16
        self.encoder4 = resnet.layer4 # out: 512, H/32
        
        # Decoder Blocks
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_4 = ConvBlock(512, 256)
        
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_3 = ConvBlock(256, 128)
        
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_2 = ConvBlock(128, 64)
        
        self.up_1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_1 = ConvBlock(64, 64)
        
        self.up_0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_0 = ConvBlock(64, 32)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x_enc0 = self.encoder0(x)     # H/4, 64
        x_enc1 = self.encoder1(x_enc0) # H/4, 64
        x_enc2 = self.encoder2(x_enc1) # H/8, 128
        x_enc3 = self.encoder3(x_enc2) # H/16, 256
        x_enc4 = self.encoder4(x_enc3) # H/32, 512
        
        # Decoder
        d4 = self.up_4(x_enc4)
        if d4.shape != x_enc3.shape:
            d4 = torch.nn.functional.interpolate(d4, size=x_enc3.shape[2:])
        d4 = torch.cat([d4, x_enc3], dim=1)
        d4 = self.dec_4(d4)
        
        d3 = self.up_3(d4)
        if d3.shape != x_enc2.shape:
            d3 = torch.nn.functional.interpolate(d3, size=x_enc2.shape[2:])
        d3 = torch.cat([d3, x_enc2], dim=1)
        d3 = self.dec_3(d3)
        
        d2 = self.up_2(d3)
        if d2.shape != x_enc1.shape:
            d2 = torch.nn.functional.interpolate(d2, size=x_enc1.shape[2:])
        d2 = torch.cat([d2, x_enc1], dim=1)
        d2 = self.dec_2(d2)
        
        # Upsample to original resolution
        d1 = self.up_1(d2)
        d1 = self.dec_1(d1) # H/2
        
        d0 = self.up_0(d1)
        if d0.shape[2:] != x.shape[2:]:
            d0 = torch.nn.functional.interpolate(d0, size=x.shape[2:])
        d0 = self.dec_0(d0) # H
        
        out = self.final(d0)
        return out
