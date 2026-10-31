import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import custom architecture and dataset
from unet_model import UNet
from wildfire_dataset import WildfireDataset

# 1. Device and Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join("checkpoints", "best_wildfire_unet.pth")

if not os.path.exists(checkpoint_path):
  raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

print("Loading trained U-Net checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = UNet(in_channels=3, out_channels=1, base_features=64).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(
    f"Loaded model from Epoch {checkpoint['epoch']} (Best Val Dice:"
    f" {checkpoint['best_dice']:.4f})"
)

# 2. Load the Unseen Validation Set
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"
val_dataset = WildfireDataset(INDEX_PATH, split="val", augment=False)
print(f"Loaded {len(val_dataset)} validation scenes.")

# 3. Find a validation sample with active wildfires to evaluate
chosen_idx = None
for i in range(len(val_dataset)):
  _, mask = val_dataset[i]
  # Check if scene contains at least 15 active fire pixels
  if mask.sum() > 15:
    chosen_idx = i
    break

if chosen_idx is None:
  chosen_idx = 0

sample_image, sample_mask = val_dataset[chosen_idx]
print(
    f"\nEvaluating Validation Sample #{chosen_idx} (Ground Truth Fire Pixels:"
    f" {int(sample_mask.sum())})..."
)

# 4. Run Model Inference
# Add batch dimension: [1, 3, 128, 256]
input_tensor = sample_image.unsqueeze(0).to(device)

with torch.no_grad():
  logits = model(input_tensor)
  # Apply Sigmoid to convert raw logits to probabilities [0.0 to 1.0]
  probs = torch.sigmoid(logits).squeeze().cpu().numpy()

# Crop padding back from (128, 256) to original Northern Algeria grid (120, 250)
# Remember: F.pad was (left=3, right=3, top=4, bottom=4)
probs_cropped = probs[4:124, 3:253]
mask_cropped = sample_mask.squeeze().cpu().numpy()[4:124, 3:253]
# Channel 2 was Delta T (denormalized for display: x * 8.0 + 10.0)
delta_t_cropped = (
    sample_image[2, 4:124, 3:253].cpu().numpy() * 8.0 + 10.0
)  # Kelvin

# Binary prediction threshold
DETECTION_THRESHOLD = 0.40
binary_pred = (probs_cropped > DETECTION_THRESHOLD).astype(np.float32)

# 5. Plot the 4-Panel Diagnostic Figure
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# Panel 1: Satellite Input (Delta T)
im0 = axes[0, 0].imshow(delta_t_cropped, cmap="inferno", origin="upper")
axes[0, 0].set_title("Input: Physical Delta T (T_3.9 - T_10.8) [Kelvin]")
fig.colorbar(im0, ax=axes[0, 0])

# Panel 2: Target Ground Truth (NASA VIIRS)
im1 = axes[0, 1].imshow(mask_cropped, cmap="gray", origin="upper")
axes[0, 1].set_title(
    f"Ground Truth Target Y (NASA Fire Pixels: {int(mask_cropped.sum())})"
)
fig.colorbar(im1, ax=axes[0, 1])

# Panel 3: Continuous U-Net Probability Map
im2 = axes[1, 0].imshow(probs_cropped, cmap="plasma", origin="upper", vmin=0, vmax=1)
axes[1, 0].set_title("U-Net Confidence Output (Sigmoid Probabilities 0.0 - 1.0)")
fig.colorbar(im2, ax=axes[1, 0])

# Panel 4: Binarized Detections (Threshold = 0.40)
im3 = axes[1, 1].imshow(binary_pred, cmap="hot", origin="upper")
axes[1, 1].set_title(
    f"Model Final Detections (Threshold = {DETECTION_THRESHOLD})"
)
fig.colorbar(im3, ax=axes[1, 1])

for ax in axes.flat:
  ax.set_xlabel("Longitude Pixel Column")
  ax.set_ylabel("Latitude Pixel Row")

plt.tight_layout()
plt.savefig("validation_prediction_preview.png", dpi=200)
print(
    "Evaluation complete! Saved diagnostic plot to"
    " 'validation_prediction_preview.png'."
)
plt.show()