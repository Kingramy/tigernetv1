import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet_model import UNet
from wildfire_day_dataset import WildfireDayDataset

# 1. Setup & Load Model
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_unet_day.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model = UNet(in_channels=4, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Load Daytime Test Scenes
test_dataset = WildfireDayDataset(INDEX_PATH, split="test", augment=False)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=0
)
print(f"Loaded {len(test_dataset)} Daytime Test Scenes.")

# 3. Cache Probability Maps
print("Generating model confidence maps on GPU...")
all_probs = []
all_targets = []

with torch.no_grad():
  for images, masks in test_loader:
    images = images.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits)

    # Crop padding back to native [120, 250]
    probs_cropped = probs[:, 0, 4:124, 3:253].cpu()
    masks_cropped = masks[:, 0, 4:124, 3:253].cpu()

    all_probs.append(probs_cropped)
    all_targets.append(masks_cropped)

probs_tensor = torch.cat(all_probs, dim=0)  # [48, 120, 250]
targets_tensor = torch.cat(all_targets, dim=0)

total_gt_fires = int(targets_tensor.sum().item())
print(f"Total active ground-truth fire pixels: {total_gt_fires:,}\n")

# 4. Sweep Thresholds from 0.50 to 0.95
thresholds = np.linspace(0.50, 0.95, 10)
results = []
eps = 1e-6

print("-" * 75)
print(
    f"{'Threshold (τ)':^14} | {'Precision':^12} | {'Recall':^12} |"
    f" {'Dice (F1)':^12} | {'Mean FP/Scene':^15}"
)
print("-" * 75)

for tau in thresholds:
  preds = (probs_tensor >= tau).float()

  p_flat = preds.view(-1)
  t_flat = targets_tensor.view(-1)

  tp = torch.sum(p_flat * t_flat).item()
  fp = torch.sum(p_flat * (1.0 - t_flat)).item()
  fn = torch.sum((1.0 - p_flat) * t_flat).item()

  precision = (tp + eps) / (tp + fp + eps)
  recall = (tp + eps) / (tp + fn + eps)
  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
  fp_per_scene = fp / len(test_dataset)

  results.append({
      "threshold": tau,
      "precision": precision,
      "recall": recall,
      "dice": dice,
      "mean_fp_per_scene": fp_per_scene,
      "tp": int(tp),
      "fp": int(fp),
      "fn": int(fn),
  })

  print(
      f"{tau:^14.2f} | {precision:^12.4f} | {recall:^12.4f} | {dice:^12.4f} |"
      f" {fp_per_scene:^15.1f}"
  )

print("-" * 75)

df_tau = pd.DataFrame(results)
best_op = df_tau.loc[df_tau["dice"].idxmax()]

print("\n==================================================")
print("OPTIMAL DAYTIME OPERATING POINT")
print(f"Optimal Threshold (τ*)   : {best_op['threshold']:.2f}")
print(f"Optimal Dice Score       : {best_op['dice']:.4f}")
print(f"Precision at τ*          : {best_op['precision']:.4f}")
print(f"Recall at τ*             : {best_op['recall']:.4f}")
print(f"False Positives / Scene  : {best_op['mean_fp_per_scene']:.1f} pixels")
print("==================================================")

# Save curve
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(
    df_tau["threshold"],
    df_tau["recall"],
    label="Recall (Detection Rate)",
    color="blue",
    linewidth=2,
    marker="o",
)
ax1.plot(
    df_tau["threshold"],
    df_tau["precision"],
    label="Precision (Alert Purity)",
    color="green",
    linewidth=2,
    marker="s",
)
ax1.plot(
    df_tau["threshold"],
    df_tau["dice"],
    label="Dice Score",
    color="red",
    linewidth=2.5,
    marker="^",
)
ax1.set_xlabel("Detection Threshold (τ)")
ax1.set_ylabel("Metric Value")
ax1.set_title("Daytime Detection Metrics vs Decision Threshold (τ)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="center left")

ax2 = ax1.twinx()
ax2.plot(
    df_tau["threshold"],
    df_tau["mean_fp_per_scene"],
    label="Mean FP / Scene",
    color="purple",
    linestyle="--",
    linewidth=2,
)
ax2.set_ylabel("False Positive Pixels per Scene (Lower is Better)")
ax2.legend(loc="center right")

plt.tight_layout()
plt.savefig("day_threshold_tuning.png", dpi=200)
print("Saved tuning curve to 'day_threshold_tuning.png'!")
plt.show()