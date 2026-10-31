import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from unet_model import UNet
from wildfire_dataset import WildfireDataset

# =====================================================================
# 1. SETUP & LOAD MODEL
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_wildfire_unet.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH, map_location=device, weights_only=False
)
model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test index records
index_df = pd.read_csv(INDEX_PATH)
test_records = index_df[index_df["split"] == "test"].copy()
test_records["hour"] = pd.to_datetime(test_records["timestamp"]).dt.hour

# Filter strictly for NIGHT scenes (UTC hour < 6 or > 18)
night_records = test_records[
    (test_records["hour"] < 6) | (test_records["hour"] > 18)
].reset_index(drop=True)
print(f"Total Nighttime Test Scenes: {len(night_records)}")

# =====================================================================
# 2. RUN INFERENCE ON NIGHT SCENES
# =====================================================================
night_probs = []
night_targets = []

with torch.no_grad():
  for idx, row in night_records.iterrows():
    data = np.load(row["file_path"])
    img = torch.from_numpy(data["image"]).float()
    mask = torch.from_numpy(data["mask"]).float().unsqueeze(0)

    # Standardize
    means = torch.tensor([295.0, 285.0, 10.0]).view(3, 1, 1)
    stds = torch.tensor([15.0, 12.0, 8.0]).view(3, 1, 1)
    img = (img - means) / stds

    # Pad (left=3, right=3, top=4, bottom=4)
    import torch.nn.functional as F

    img_padded = F.pad(img, (3, 3, 4, 4), mode="reflect").unsqueeze(0).to(device)

    logits = model(img_padded)
    probs = torch.sigmoid(logits)

    # Crop padding back to native [120, 250]
    probs_cropped = probs[0, 0, 4:124, 3:253].cpu()

    night_probs.append(probs_cropped)
    night_targets.append(mask[0])

# Concatenate all night pixels
all_probs = torch.stack(night_probs)
all_targets = torch.stack(night_targets)

total_fire_px = int(all_targets.sum().item())
total_px = all_targets.numel()
print(
    f"Evaluated {total_px:,} nighttime pixels. Active fire pixels:"
    f" {total_fire_px:,} ({total_fire_px/total_px*100:.3f}%)\n"
)

# =====================================================================
# 3. THRESHOLD SWEEP FOR NIGHT SCENES
# =====================================================================
thresholds = np.linspace(0.10, 0.90, 17)
rows = []
eps = 1e-6

print("-" * 65)
print(
    f"{'Threshold (τ)':^14} | {'Precision':^12} | {'Recall':^12} |"
    f" {'Dice (F1)':^12} | {'IoU':^10}"
)
print("-" * 65)

for tau in thresholds:
  preds = (all_probs >= tau).float()

  p_flat = preds.view(-1)
  t_flat = all_targets.view(-1)

  tp = torch.sum(p_flat * t_flat).item()
  fp = torch.sum(p_flat * (1.0 - t_flat)).item()
  fn = torch.sum((1.0 - p_flat) * t_flat).item()

  precision = (tp + eps) / (tp + fp + eps)
  recall = (tp + eps) / (tp + fn + eps)
  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
  iou = (tp + eps) / (tp + fp + fn + eps)

  rows.append({
      "threshold": tau,
      "precision": precision,
      "recall": recall,
      "dice": dice,
      "iou": iou,
      "tp": int(tp),
      "fp": int(fp),
      "fn": int(fn),
  })

  print(
      f"{tau:^14.2f} | {precision:^12.4f} | {recall:^12.4f} | {dice:^12.4f} |"
      f" {iou:^10.4f}"
  )

print("-" * 65)

df_night = pd.DataFrame(rows)
best_night = df_night.loc[df_night["dice"].idxmax()]

print("\n==================================================")
print("NIGHTTIME BASELINE RESULTS")
print(f"Optimal Threshold (τ*) : {best_night['threshold']:.2f}")
print(f"Optimal Dice (F1) Score: {best_night['dice']:.4f}")
print(f"Precision at τ*        : {best_night['precision']:.4f}")
print(f"Recall at τ*           : {best_night['recall']:.4f}")
print(f"IoU at τ*              : {best_night['iou']:.4f}")
print(
    f"True Positives: {int(best_night['tp'])}, False Positives:"
    f" {int(best_night['fp'])}, Missed Fires: {int(best_night['fn'])}"
)
print("==================================================")