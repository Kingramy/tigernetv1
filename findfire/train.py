import os
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import our custom modules
from losses_and_metrics import FocalTverskyLoss, calculate_segmentation_metrics
from unet_model import UNet
from wildfire_dataset import WildfireDataset

# =====================================================================
# 1. HYPERPARAMETERS & SYSTEM SETUP
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# Select execution device (GTX 1660 SUPER)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==================================================")
print(f"Executing Training Pipeline on: {device}")
if device.type == "cuda":
  print(f"Active GPU: {torch.cuda.get_device_name(0)}")
print("==================================================\n")

# =====================================================================
# 2. DATA LOADERS
# =====================================================================
print("Loading Datasets...")
train_dataset = WildfireDataset(INDEX_PATH, split="train", augment=True)
val_dataset = WildfireDataset(INDEX_PATH, split="val", augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(
    f"Training batches per epoch   : {len(train_loader)} ({len(train_dataset)} samples)"
)
print(
    f"Validation batches per epoch : {len(val_loader)} ({len(val_dataset)} samples)\n"
)

# =====================================================================
# 3. MODEL, LOSS, OPTIMIZER & SCHEDULER
# =====================================================================
model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0)
optimizer = AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# =====================================================================
# 4. TRAINING & VALIDATION ENGINE
# =====================================================================
best_val_dice = 0.0
best_model_path = os.path.join(CHECKPOINT_DIR, "best_wildfire_unet.pth")

print("Starting Model Training...\n")
total_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
  epoch_start = time.time()

  # -------------------------------------------------------------
  # A. TRAINING PHASE
  # -------------------------------------------------------------
  model.train()
  running_train_loss = 0.0

  for images, masks in train_loader:
    images = images.to(device)
    masks = masks.to(device)

    # 1. Forward pass
    logits = model(images)

    # 2. Compute loss
    loss = criterion(logits, masks)

    # 3. Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_train_loss += loss.item()

  train_epoch_loss = running_train_loss / len(train_loader)

  # -------------------------------------------------------------
  # B. VALIDATION PHASE
  # -------------------------------------------------------------
  model.eval()
  running_val_loss = 0.0
  val_dice_list = []
  val_iou_list = []
  val_recall_list = []

  with torch.no_grad():
    for images, masks in val_loader:
      images = images.to(device)
      masks = masks.to(device)

      logits = model(images)
      val_loss = criterion(logits, masks)
      running_val_loss += val_loss.item()

      # Compute pixel metrics on binarized predictions
      metrics = calculate_segmentation_metrics(logits, masks, threshold=0.5)
      val_dice_list.append(metrics["dice"])
      val_iou_list.append(metrics["iou"])
      val_recall_list.append(metrics["recall"])

  val_epoch_loss = running_val_loss / len(val_loader)
  mean_val_dice = np.mean(val_dice_list)
  mean_val_iou = np.mean(val_iou_list)
  mean_val_recall = np.mean(val_recall_list)

  # Update learning rate schedule
  scheduler.step()
  current_lr = scheduler.get_last_lr()[0]
  epoch_duration = time.time() - epoch_start

  # -------------------------------------------------------------
  # C. CONSOLE LOGGING & CHECKPOINTING
  # -------------------------------------------------------------
  saved_flag = ""
  if mean_val_dice > best_val_dice:
    best_val_dice = mean_val_dice
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_val_dice,
            "val_loss": val_epoch_loss,
        },
        best_model_path,
    )
    saved_flag = "--> [SAVED BEST MODEL]"

  print(
      f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_duration:.1f}s) | LR:"
      f" {current_lr:.6f} | "
      f"Train Loss: {train_epoch_loss:.4f} | "
      f"Val Loss: {val_epoch_loss:.4f} | "
      f"Val Dice: {mean_val_dice:.4f} | "
      f"Val Recall: {mean_val_recall:.4f} {saved_flag}"
  )

total_elapsed = (time.time() - total_start_time) / 60
print("\n==================================================")
print("TRAINING PROCESS COMPLETED!")
print(f"Total Time Elapsed    : {total_elapsed:.2f} minutes")
print(f"Best Validation Dice  : {best_val_dice:.4f}")
print(f"Best Model Checkpoint : {best_model_path}")
print("==================================================")