import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from ecg_digitizer.modules.stage05_models import UNetPlusPlusResNet18
from mlflow_config import setup_mlflow, log_dataset_lineage, log_standard_metrics
import mlflow

class ECGDataset(Dataset):
    def __init__(self, data_dirs):
        self.samples = []
        for d in data_dirs:
            if not os.path.exists(d):
                continue
            imgs = glob.glob(os.path.join(d, "*_img.png"))
            for img_path in imgs:
                mask_path = img_path.replace("_img.png", "_mask.png")
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        h, w = img.shape
        # Pad to multiple of 32 (only bottom and right borders to maintain 1:1 local coordinate layout)
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        img_tensor = torch.from_numpy(img_padded).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask_padded).float().unsqueeze(0) / 255.0
        
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
    print("Initializing Stage 5 Training...")
    
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    setup_mlflow("ECG_Stage5_Training")
    
    with mlflow.start_run():
        log_dataset_lineage(
            dataset_version="v1.0-scratch",
            stage4_version="v1.0",
            stage5_1_version="resnet18-unet-scratch",
            stage5_2_version="consensus-v1",
            biomarker_version="v1",
            calibration_version="v1",
            patient_split_version="v1"
        )
        
        mlflow.log_params({
            "model": "UNetPlusPlusResNet18",
            "epochs": 3,
            "learning_rate": 1e-3,
            "batch_size": 1
        })
        
        train_dirs = ["pseudo_labels/Copy of 19_male"]
        val_dirs = ["pseudo_labels/Copy of 20_male"]
        
        # Load dataset
        train_dataset = ECGDataset(train_dirs)
        val_dataset = ECGDataset(val_dirs)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Warning: Training or Validation datasets are empty! Check pseudo_labels directory.")
            
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetPlusPlusResNet18(in_channels=1, out_channels=1).to(device)
        
        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_iou = -1.0
        epochs = 3
        
        history = {
            "train_loss": [], "val_loss": [], 
            "val_iou": [], "val_dice": [], 
            "val_precision": [], "val_recall": []
        }
        
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
                
            train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            
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
                    
            n_val = len(val_loader) if len(val_loader) > 0 else 1
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
                torch.save(model.state_dict(), "stage5_best_model.pth")
                
            log_standard_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "precision": val_prec,
                "recall": val_rec,
                "iou": val_iou,
                "dice": val_dice
            })
            
        with open("stage5_training_history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        mlflow.log_artifact("stage5_best_model.pth")
        mlflow.log_artifact("stage5_training_history.json")
        
    print("Training complete. Best model saved to stage5_best_model.pth")

if __name__ == "__main__":
    train_model()
