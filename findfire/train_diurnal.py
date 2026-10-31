import os
import time
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from losses_and_metrics import calculate_segmentation_metrics
from unet_model import UNet
from wildfire_dataset import WildfireDataset

# =====================================================================
# 1. ADAPTIVE DIURNAL LOSS FUNCTION
# =====================================================================


class AdaptiveDiurnalFocalTverskyLoss(torch.nn.Module):
  """Adapts alpha and beta per sample based on the Solar Cycle value S(t).

  - Day (S >= 0.5): alpha=0.6, beta=0.4 (suppresses false alarms on hot rock)
  - Night (S < 0.5): alpha=0.3, beta=0.7 (sensitive to faint fires in the dark)
  """

  def __init__(self, gamma=4.0 / 3.0, eps=1e-6):
    super(AdaptiveDiurnalFocalTverskyLoss, self).__init__()
    self.gamma = gamma
    self.eps = eps

  def forward(self, logits, targets, solar_vals):
    probs = torch.sigmoid(logits)
    batch_size = logits.size(0)
    loss_sum = 0.0

    probs_flat = probs.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)

    for b in range(batch_size):
      p = probs_flat[b]
      g = targets_flat[b]
      s = solar_vals[b].item()

      # Adaptive weights
      if s >= 0.5:
        alpha, beta = 0.6, 0.4  # Daytime penalization of false alarms
      else:
        alpha, beta = 0.3, 0.7  # Nighttime sensitivity to small fires

      tp = torch.sum(p * g)
      fp = torch.sum(p * (1.0 - g))
      fn = torch.sum((1.0 - p) * g)

      ti = (tp + self.eps) / (tp + alpha * fp + beta * fn + self.eps)
      loss_sum += torch.pow((1.0 - ti), (1.0 / self.gamma))

    return loss_sum / batch_size


# =====================================================================
# 2. HYPERPARAMETERS & SETUP
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
print(f"Training Diurnal 4-Channel U-Net on: {device}")
if device.type == "cuda":
  print(f"GPU: {torch.cuda.get_device_name(0)}")
print("==================================================\n")

# Datasets
train_dataset = WildfireDataset(INDEX_PATH, split="train", augment=True)
val_dataset = WildfireDataset(INDEX_PATH, split="val", augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# 4-Channel Model
model = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
criterion = AdaptiveDiurnalFocalTverskyLoss()
optimizer = AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_dice = 0.0
best_model_path = os.path.join(CHECKPOINT_DIR, "best_diurnal_unet.pth")

print("Beginning 4-Channel Model Training...\n")

for epoch in range(1, EPOCHS + 1):
  epoch_start = time.time()

  # Train
  model.train()
  train_loss_acc = 0.0

  for images, masks, solar_vals in train_loader:
    images = images.to(device)
    masks = masks.to(device)
    solar_vals = solar_vals.to(device)

    logits = model(images)
    loss = criterion(logits, masks, solar_vals)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_acc += loss.item()

  train_epoch_loss = train_loss_acc / len(train_loader)

  # Validation
  model.eval()
  val_loss_acc = 0.0
  val_dice_list, val_recall_list = [], []

  with torch.no_grad():
    for images, masks, solar_vals in val_loader:
      images = images.to(device)
      masks = masks.to(device)
      solar_vals = solar_vals.to(device)

      logits = model(images)
      loss = criterion(logits, masks, solar_vals)
      val_loss_acc += loss.item()

      m = calculate_segmentation_metrics(logits, masks, threshold=0.5)
      val_dice_list.append(m["dice"])
      val_recall_list.append(m["recall"])

  val_epoch_loss = val_loss_acc / len(val_loader)
  mean_val_dice = np.mean(val_dice_list)
  mean_val_recall = np.mean(val_recall_list)

  scheduler.step()
  current_lr = scheduler.get_last_lr()[0]
  epoch_time = time.time() - epoch_start

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
    saved_flag = "--> [SAVED BEST MODEL]"

  print(
      f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({epoch_time:.1f}s) | LR:"
      f" {current_lr:.6f} | "
      f"Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | "
      f"Val Dice: {mean_val_dice:.4f} | Val Recall:"
      f" {mean_val_recall:.4f} {saved_flag}"
  )

print("\n==================================================")
print(f"RETRAINING COMPLETE! Best Model Checkpoint: {best_model_path}")
print(f"Best Validation Dice: {best_val_dice:.4f}")
print("==================================================")