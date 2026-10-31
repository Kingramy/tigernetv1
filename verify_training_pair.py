import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from satpy import Scene
from pyresample import create_area_def

# 1. Locate the newly downloaded July 21 .nat file
nat_candidates = glob.glob("*20260721*.nat")
if not nat_candidates:
    raise FileNotFoundError("Could not find the extracted July 21 .nat file. Make sure the .zip is extracted!")

target_nat_file = nat_candidates[0]
print(f"Loading satellite scan: {target_nat_file}")

# 2. Load the scene and the thermal bands
scn = Scene(filenames=[target_nat_file], reader="seviri_l1b_native")
scn.load(["IR_039", "IR_108"])

# 3. Define the Northern Algeria bounding box (120 x 250)
algeria_extent = [-2.0, 34.0, 8.5, 37.5]
algeria_shape = (120, 250)

algeria_area = create_area_def(
    area_id="algeria_north",
    projection={"proj": "latlong", "datum": "WGS84"},
    area_extent=algeria_extent,
    shape=algeria_shape,
    units="degrees",
)

# 4. Resample to our Northern Algeria grid
print("Cropping and resampling to Northern Algeria...")
local_scn = scn.resample(algeria_area, resampler="nearest")

t_39 = local_scn["IR_039"].values
t_108 = local_scn["IR_108"].values
delta_t = t_39 - t_108

# 5. Load our NASA Ground Truth Mask
mask_file = "sample_ground_truth_mask.npy"
if not os.path.exists(mask_file):
    raise FileNotFoundError(f"Missing {mask_file}. Run create_mask.py first!")

ground_truth_mask = np.load(mask_file)

# 6. Verify shapes match perfectly
print(f"\nSatellite Tensor Shape: {delta_t.shape}")
print(f"Ground Truth Mask Shape: {ground_truth_mask.shape}")
assert delta_t.shape == ground_truth_mask.shape, "Shape mismatch between input and target!"

# 7. Visualize the Training Pair Side-by-Side with Overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot A: The Physical Thermal Signal (Delta T)
im0 = axes[0].imshow(delta_t, cmap="inferno", origin="upper")
axes[0].set_title("Input X: Delta T (T_3.9 - T_10.8)")
fig.colorbar(im0, ax=axes[0], label="Difference (Kelvin)")

# Plot B: The Target Ground Truth Mask (Y)
im1 = axes[1].imshow(ground_truth_mask, cmap="gray", origin="upper")
axes[1].set_title("Target Y: NASA VIIRS Binary Mask")
fig.colorbar(im1, ax=axes[1], label="0 = Safe, 1 = Fire")

# Plot C: Direct Overlay (Ground truth plotted over Delta T)
axes[2].imshow(delta_t, cmap="inferno", origin="upper")
# Find fire coordinates in the mask and plot them as cyan markers
fire_y, fire_x = np.where(ground_truth_mask == 1.0)
axes[2].scatter(fire_x, fire_y, color="cyan", s=15, marker="x", label="NASA Fire Coordinates")
axes[2].set_title("Alignment Verification: Target Points Over Heatmap")
axes[2].legend(loc="lower left")

for ax in axes:
    ax.set_xlabel("Pixel Column (Longitude)")
    ax.set_ylabel("Pixel Row (Latitude)")

plt.tight_layout()
plt.savefig("alignment_verification.png", dpi=200)
print("\nVerification plot saved as 'alignment_verification.png'!")
plt.show()