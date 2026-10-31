import os
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from losses_and_metrics import calculate_segmentation_metrics
from losses_day import BalancedDayFireLoss
from unet_model import UNet
from wildfire_day_dataset import WildfireDayDataset

# =====================================================================
# 1. SETUP & CONFIGURATION
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==================================================")
print(f"Training Daytime U-Net (UNet_Day) on: {device}")
if device.type == "cuda":
  print(f"GPU: {torch.cuda.get_device_name(0)}")
print("==================================================\n")

# Datasets
train_dataset = WildfireDayDataset(INDEX_PATH, split="train", augment=True)
val_dataset = WildfireDayDataset(INDEX_PATH, split="val", augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(
    f"Daytime Train samples: {len(train_dataset)} ({len(train_loader)} batches)"
)
print(
    f"Daytime Val samples  : {len(val_dataset)} ({len(val_loader)} batches)\n"
)

# Initialize UNet with in_channels=4
model = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
criterion = BalancedDayFireLoss(alpha=0.5, beta=0.5, gamma=4.0 / 3.0)
optimizer = AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_dice = 0.0
best_model_path = os.path.join(CHECKPOINT_DIR, "best_unet_day.pth")

print("Beginning Training for UNet_Day...\n")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
  epoch_start = time.time()

  # Train
  model.train()
  train_loss_sum = 0.0

  for images, masks in train_loader:
    images = images.to(device)
    masks = masks.to(device)

    logits = model(images)
    loss = criterion(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_sum += loss.item()

  train_epoch_loss = train_loss_sum / len(train_loader)

  # Validate
  model.eval()
  val_loss_sum = 0.0
  val_dice_list, val_recall_list = [], []

  with torch.no_grad():
    for images, masks in val_loader:
      images = images.to(device)
      masks = masks.to(device)

      logits = model(images)
      loss = criterion(logits, masks)
      val_loss_sum += loss.item()

      m = calculate_segmentation_metrics(logits, masks, threshold=0.5)
      val_dice_list.append(m["dice"])
      val_recall_list.append(m["recall"])

  val_epoch_loss = val_loss_sum / len(val_loader)
  mean_val_dice = np.mean(val_dice_list)
  mean_val_recall = np.mean(val_recall_list)

  scheduler.step()
  current_lr = scheduler.get_last_lr()[0]
  duration = time.time() - epoch_start

  saved_flag = ""
  if mean_val_dice > best_val_dice:
    best_val_dice = mean_val_dice
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_dice": best_val_dice,
            "val_loss": val_epoch_loss,
        },
        best_model_path,
    )
    saved_flag = "--> [SAVED BEST DAY MODEL]"

  print(
      f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({duration:.1f}s) | LR: {current_lr:.6f}"
      f" | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} |"
      f" Val Dice: {mean_val_dice:.4f} | Val Recall: {mean_val_recall:.4f}"
      f" {saved_flag}"
  )

total_mins = (time.time() - start_time) / 60
print("\n==================================================")
print("UNet_Day TRAINING COMPLETE!")
print(f"Elapsed Time         : {total_mins:.2f} minutes")
print(f"Best Val Dice        : {best_val_dice:.4f}")
print(f"Saved Checkpoint     : {best_model_path}")
print("==================================================")