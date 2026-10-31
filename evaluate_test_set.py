import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from unet_model import UNet
from wildfire_dataset import WildfireDataset

# =====================================================================
# 1. SETUP & MODEL INITIALIZATION
# =====================================================================
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
CHECKPOINT_PATH = os.path.join("checkpoints", "best_wildfire_unet.pth")

if not os.path.exists(CHECKPOINT_PATH):
  raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("==================================================")
print(f"Executing Test Set Evaluation on: {device}")
if device.type == "cuda":
  print(f"GPU: {torch.cuda.get_device_name(0)}")
print("==================================================\n")

# Load Checkpoint with PyTorch 2.6 security parameter
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(
    f"Loaded trained model from Epoch {checkpoint['epoch']} (Best Val Dice:"
    f" {checkpoint['best_dice']:.4f})\n"
)

# =====================================================================
# 2. LOAD THE UNTOUCHED TEST DATASET
# =====================================================================
test_dataset = WildfireDataset(INDEX_PATH, split="test", augment=False)
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=0
)

print(f"Total Test Scenes to Evaluate: {len(test_dataset)}")
print(f"Total Batches                : {len(test_loader)}\n")

# =====================================================================
# 3. RUN INFERENCE & CACHE PROBABILITY MAPS
# =====================================================================
print("Running forward passes across all test scenes...")
all_probs = []
all_targets = []

with torch.no_grad():
  for images, masks in test_loader:
    images = images.to(device)
    # Forward pass -> Sigmoid probabilities
    logits = model(images)
    probs = torch.sigmoid(logits)

    # Crop symmetric padding back to native Northern Algeria grid (120 x 250)
    # Padding applied was: left=3, right=3, top=4, bottom=4
    probs_cropped = probs[:, :, 4:124, 3:253]
    masks_cropped = masks[:, :, 4:124, 3:253]

    # Collect predictions and targets on CPU
    all_probs.append(probs_cropped.cpu())
    all_targets.append(masks_cropped.cpu())

# Concatenate into full test dataset tensors: [89, 1, 120, 250]
all_probs = torch.cat(all_probs, dim=0)
all_targets = torch.cat(all_targets, dim=0)

total_fire_pixels = int(all_targets.sum().item())
total_pixels_evaluated = all_targets.numel()
print(
    f"Inference complete! Evaluated {total_pixels_evaluated:,} total pixels."
)
print(
    f"Total active ground-truth fire pixels in test set: {total_fire_pixels:,}"
    f" ({total_fire_pixels / total_pixels_evaluated * 100:.3f}%)\n"
)

# =====================================================================
# 4. OPTIMAL THRESHOLD SEARCH (τ ∈ [0.10, 0.90])
# =====================================================================
thresholds = np.linspace(0.10, 0.90, 17)
results = []

print("Evaluating detection thresholds...")
print("-" * 68)
print(
    f"{'Threshold (τ)':^14} | {'Precision':^12} | {'Recall':^12} |"
    f" {'Dice (F1)':^12} | {'IoU':^10}"
)
print("-" * 68)

eps = 1e-6

for tau in thresholds:
  # Binarize predictions according to threshold tau
  preds = (all_probs >= tau).float()

  # Flatten all pixels across the entire test set
  preds_flat = preds.view(-1)
  targets_flat = all_targets.view(-1)

  # Compute dataset-level Confusion Matrix elements
  tp = torch.sum(preds_flat * targets_flat).item()
  fp = torch.sum(preds_flat * (1.0 - targets_flat)).item()
  fn = torch.sum((1.0 - preds_flat) * targets_flat).item()

  # Compute standard segmentation metrics
  precision = (tp + eps) / (tp + fp + eps)
  recall = (tp + eps) / (tp + fn + eps)
  dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
  iou = (tp + eps) / (tp + fp + fn + eps)

  results.append({
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

print("-" * 68)

results_df = pd.DataFrame(results)

# Find threshold that maximizes Dice score (F1)
best_row = results_df.loc[results_df["dice"].idxmax()]
best_tau = best_row["threshold"]

print("\n==================================================")
print("OPTIMAL THRESHOLD IDENTIFIED")
print(f"Optimal Threshold (τ*) : {best_tau:.2f}")
print(f"Optimal Dice (F1) Score: {best_row['dice']:.4f}")
print(f"Precision at τ*        : {best_row['precision']:.4f}")
print(f"Recall at τ*           : {best_row['recall']:.4f}")
print(f"Intersection over Union: {best_row['iou']:.4f}")
print(
    f"True Positives: {int(best_row['tp'])}, False Positives:"
    f" {int(best_row['fp'])}, Missed Fires: {int(best_row['fn'])}"
)
print("==================================================\n")

# Save metric evaluations to CSV
results_df.to_csv("test_set_threshold_metrics.csv", index=False)
print("Saved evaluation table to 'test_set_threshold_metrics.csv'.")

# =====================================================================
# 5. PUBLICATION-QUALITY VISUALIZATIONS
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Metrics vs Threshold Curve
axes[0].plot(
    results_df["threshold"],
    results_df["recall"],
    label="Recall (Detection Coverage)",
    color="blue",
    linewidth=2,
    marker="o",
)
axes[0].plot(
    results_df["threshold"],
    results_df["precision"],
    label="Precision (Alert Accuracy)",
    color="green",
    linewidth=2,
    marker="s",
)
axes[0].plot(
    results_df["threshold"],
    results_df["dice"],
    label="Dice Score (F1-Harmonic)",
    color="red",
    linewidth=2.5,
    marker="^",
)
axes[0].axvline(
    best_tau,
    color="black",
    linestyle="--",
    alpha=0.8,
    label=f"Optimal τ* = {best_tau:.2f}",
)
axes[0].set_xlabel("Detection Threshold (τ)", fontsize=12)
axes[0].set_ylabel("Metric Value (0.0 to 1.0)", fontsize=12)
axes[0].set_title(
    "Wildfire Detection Metrics vs. Decision Threshold", fontsize=13
)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(fontsize=10)

# Subplot 2: Precision-Recall (PR) Curve
axes[1].plot(
    results_df["recall"],
    results_df["precision"],
    color="darkorange",
    linewidth=2.5,
    marker="D",
)
axes[1].scatter(
    best_row["recall"],
    best_row["precision"],
    color="red",
    s=120,
    zorder=5,
    label=f"Optimal Operating Point (τ*={best_tau:.2f})",
)
axes[1].set_xlabel("Recall (Sensitivity)", fontsize=12)
axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
axes[1].set_title(
    "Precision-Recall (PR) Operational Curve", fontsize=13
)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("test_evaluation_curves.png", dpi=200)
print("Saved performance curves to 'test_evaluation_curves.png'!")
plt.show()