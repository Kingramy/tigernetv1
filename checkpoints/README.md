# Model Checkpoints

This directory is used to store trained PyTorch model checkpoints (`.pth`).

Due to GitHub's 100 MB file size limit, large checkpoint weights are excluded from version control:
- `best_wildfire_unet.pth` (~372 MB)
- `best_unet_day_800.pth` (~124 MB)
- `best_unet_night_800.pth` (~124 MB)
- `best_diurnal_unet.pth` (~124 MB)
- `best_unet_day.pth` (~124 MB)

To regenerate these models:
1. Run `python train_day_800.py` for the daytime regime.
2. Run `python train_night_800.py` for the nocturnal regime.

