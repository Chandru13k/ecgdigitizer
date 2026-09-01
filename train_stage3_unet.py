import os
import cv2
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from ecg_digitizer.modules.stage03_models import Stage3UNet
from mlflow_config import setup_mlflow, log_dataset_lineage, log_standard_metrics
import mlflow

class SyntheticSeparatorDataset(Dataset):
    """
    Dataset that dynamically generates synthetic training patches by taking
    clean, separator-free ECG panels, converting them to binary (white on black),
    and overlaying randomized vertical lines.
    """
    def __init__(self, clean_img_paths, patch_size=256, num_samples=200):
        self.clean_img_paths = [p for p in clean_img_paths if os.path.exists(p)]
        self.patch_size = patch_size
        self.num_samples = num_samples
        
        # Preload and binarize images (white trace/foreground on black background)
        self.images = []
        for path in self.clean_img_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Convert to binary: white trace (255), black background (0)
                binary = (img < 180).astype(np.uint8) * 255
                self.images.append(binary)
                
        if not self.images:
            print("Warning: No clean ECG images found. Generating synthetic mock images.")
            mock_img = np.zeros((1000, 2000), dtype=np.uint8)
            for row in range(4):
                y_mid = 120 + row * 250
                pts = [int(y_mid + 40 * np.sin(x * 0.03)) for x in range(2000)]
                for x in range(1, 2000):
                    cv2.line(mock_img, (x-1, pts[x-1]), (x, pts[x]), 255, 2)
            self.images.append(mock_img)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        base_img = random.choice(self.images)
        h, w = base_img.shape
        
        y1 = random.randint(0, h - self.patch_size)
        x1 = random.randint(0, w - self.patch_size)
        patch = base_img[y1:y1+self.patch_size, x1:x1+self.patch_size].copy()
        
        # 40% chance to completely clear waveform traces to train on clean backgrounds
        if random.random() < 0.4:
            patch = np.zeros_like(patch)
            
        mask = np.zeros_like(patch, dtype=np.uint8)
        
        if random.random() < 0.7:
            num_seps = 1
            for _ in range(num_seps):
                sep_x = random.randint(10, self.patch_size - 10)
                if random.random() < 0.8:
                    y_start, y_end = 0, self.patch_size
                else:
                    height = random.randint(int(self.patch_size * 0.1), int(self.patch_size * 0.3))
                    y_start = random.randint(0, self.patch_size - height)
                    y_end = y_start + height
                    
                thickness = random.randint(1, 4)
                # Draw synthetic separator as white (255) on black background
                if thickness == 1:
                    dx_min, dx_max = 0, 0
                elif thickness == 2:
                    dx_min, dx_max = 0, 1
                elif thickness == 3:
                    dx_min, dx_max = -1, 1
                else:
                    dx_min, dx_max = -1, 2
                for y_idx in range(y_start, y_end):
                    for dx in range(dx_min, dx_max + 1):
                        curr_x = sep_x + dx
                        if 0 <= curr_x < self.patch_size:
                            mask[y_idx, curr_x] = 255
                            patch[y_idx, curr_x] = 255

        # Tensors: 0.0 to 1.0 representation
        img_tensor = torch.from_numpy(patch).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return img_tensor, mask_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def calculate_metrics(probs, targets, threshold=0.5):
    preds = (probs > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    return precision, recall, iou, dice

def train_model():
    print("Initializing Stage 3 U-Net Training...")
    
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage3_Separator_Segmentation")
    
    # Locate clean ECG panels to use as templates
    clean_images = glob.glob("stage3_output/*/05_clean_ecg.png")
    if not clean_images:
        # Fallback to active panels if stage3 outputs aren't available yet
        clean_images = glob.glob("stage2_2_output/*/01_active_panel.png")
        
    print(f"Using {len(clean_images)} clean image panels for synthetic training data.")
    
    # Split into train/validation images
    random.seed(42)
    random.shuffle(clean_images)
    split = max(1, int(len(clean_images) * 0.8))
    train_paths = clean_images[:split]
    val_paths = clean_images[split:]
    
    train_dataset = SyntheticSeparatorDataset(train_paths, patch_size=128, num_samples=600)
    val_dataset = SyntheticSeparatorDataset(val_paths, patch_size=128, num_samples=200)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cpu")  # CPU is faster/more reliable in this environment
    model = Stage3UNet(in_channels=1, out_channels=1).to(device)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_iou = -1.0
    epochs = 12
    
    history = {
        "train_loss": [], "val_loss": [], 
        "val_iou": [], "val_dice": [], 
        "val_precision": [], "val_recall": []
    }
    
    with mlflow.start_run():
        log_dataset_lineage(
            dataset_version="v1.0-synthetic-separators",
            stage4_version="n/a",
            stage5_1_version="n/a",
            stage5_2_version="n/a",
            biomarker_version="n/a",
            calibration_version="v1",
            patient_split_version="v1"
        )
        
        mlflow.log_params({
            "model": "Stage3UNet",
            "epochs": epochs,
            "learning_rate": 1e-3,
            "batch_size": 8,
            "patch_size": 128
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0
            val_prec, val_rec, val_iou, val_dice = 0, 0, 0, 0
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    p, r, i, d = calculate_metrics(probs, masks)
                    val_prec += p
                    val_rec += r
                    val_iou += i
                    val_dice += d
                    
            n_val = len(val_loader)
            val_loss /= n_val
            val_prec /= n_val
            val_rec /= n_val
            val_iou /= n_val
            val_dice /= n_val
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} "
                  f"| Val Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_precision"].append(val_prec)
            history["val_recall"].append(val_rec)
            history["val_iou"].append(val_iou)
            history["val_dice"].append(val_dice)
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), "stage3_unet_best.pth")
                print("  Saved best model to stage3_unet_best.pth")
                
            log_standard_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "precision": val_prec,
                "recall": val_rec,
                "iou": val_iou,
                "dice": val_dice
            })
            
        with open("stage3_training_history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        mlflow.log_artifact("stage3_unet_best.pth")
        mlflow.log_artifact("stage3_training_history.json")
        
    print("Training complete. Model saved as stage3_unet_best.pth")

if __name__ == "__main__":
    train_model()
