import os
import torch
from torch.utils.data import DataLoader
from wildfire_dataset import WildfireDataset

# Path to the dataset index created by build_dataset.py
INDEX_PATH = r"D:\FINDFIRE_DATA\processed\dataset_index.csv"

if not os.path.exists(INDEX_PATH):
  raise FileNotFoundError(
      f"Index file not found at {INDEX_PATH}. Ensure build_dataset.py has"
      " finished!"
  )

print("Initializing PyTorch Datasets...")
train_dataset = WildfireDataset(INDEX_PATH, split="train", augment=True)
val_dataset = WildfireDataset(INDEX_PATH, split="val", augment=False)
test_dataset = WildfireDataset(INDEX_PATH, split="test", augment=False)

print(f"Train samples     : {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples      : {len(test_dataset)}")

# Build DataLoader
# batch_size=8 is well-suited for a 6GB VRAM GPU (uses ~15MB of VRAM per batch)
BATCH_SIZE = 8
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# Fetch the first mini-batch
batch_images, batch_masks = next(iter(train_loader))

print("\n--- Mini-Batch Verification ---")
print(
    f"Batch Images Tensor Shape [B, C, H, W]: {batch_images.shape} | Dtype:"
    f" {batch_images.dtype}"
)
print(
    f"Batch Masks Tensor Shape  [B, C, H, W]: {batch_masks.shape}  | Dtype:"
    f" {batch_masks.dtype}"
)
print(f"Image Normalized Min Value            : {batch_images.min():.2f}")
print(f"Image Normalized Max Value            : {batch_images.max():.2f}")
print(
    f"Mask Unique Values                    :"
    f" {torch.unique(batch_masks).tolist()}"
)

# Verify dimensions match U-Net requirements
assert batch_images.shape == (
    BATCH_SIZE,
    3,
    128,
    256,
), "Unexpected image shape!"
assert batch_masks.shape == (
    BATCH_SIZE,
    1,
    128,
    256,
), "Unexpected mask shape!"
print(
    "\nVerification Successful: Tensors are aligned and ready for the U-Net"
    " architecture."
)