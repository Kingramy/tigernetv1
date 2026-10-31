import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from satpy import Scene
from pyresample import create_area_def

# 1. Locate the raw satellite file
raw_data_dir = "."  # Current directory, adjust if files are in another folder
nat_files = glob.glob(os.path.join(raw_data_dir, "*.nat"))
if not nat_files:
    raise FileNotFoundError("No .nat file found in the directory.")

target_file = nat_files[0]
print(f"Opening: {target_file}")

# 2. Load the scene and the required thermal bands
scn = Scene(filenames=[target_file], reader="seviri_l1b_native")
scn.load(["IR_039", "IR_108"])

# 3. Define the geographic bounding box for Northern Algeria
# Extent format: [min_longitude, min_latitude, max_longitude, max_latitude]
# Shape format: (height_in_pixels, width_in_pixels)
algeria_extent = [-2.0, 34.0, 8.5, 37.5]
algeria_shape = (120, 250)

print("\nDefining Northern Algeria projection area...")
algeria_area = create_area_def(
    area_id="algeria_north",
    projection={"proj": "latlong", "datum": "WGS84"},
    area_extent=algeria_extent,
    shape=algeria_shape,
    units="degrees",
)

# 4. Resample (crop and project) the raw satellite view onto our target grid
print("Resampling satellite image to Northern Algeria grid (120 x 250)...")
local_scn = scn.resample(algeria_area, resampler="nearest")

# 5. Extract raw numpy arrays
t_39 = local_scn["IR_039"].values
t_108 = local_scn["IR_108"].values

print(f"\nCropped T_3.9 Array Shape: {t_39.shape}")
print(f"Cropped T_10.8 Array Shape: {t_108.shape}")

# 6. Compute the physical Differential Feature: Delta T = T_3.9 - T_10.8
delta_t = t_39 - t_108

print(f"\nDelta T Min: {np.nanmin(delta_t):.2f} K")
print(f"Delta T Max: {np.nanmax(delta_t):.2f} K")
print(f"Delta T Mean: {np.nanmean(delta_t):.2f} K")

# 7. Stack into a 3-channel input tensor ready for a neural network: [3, 120, 250]
input_tensor_3ch = np.stack([t_39, t_108, delta_t], axis=0)
print(f"\nFinal Network Input Tensor Shape [Channels, Height, Width]: {input_tensor_3ch.shape}")

# 8. Plot and save an image so you can inspect the cropped map visually
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].imshow(t_39, cmap="inferno", origin="upper")
axes[0].set_title("IR 3.9 um (Mid-Wave IR)")
fig.colorbar(im0, ax=axes[0], label="Kelvin")

im1 = axes[1].imshow(t_108, cmap="plasma", origin="upper")
axes[1].set_title("IR 10.8 um (Thermal IR)")
fig.colorbar(im1, ax=axes[1], label="Kelvin")

im2 = axes[2].imshow(delta_t, cmap="magma", origin="upper")
axes[2].set_title("Delta T (3.9um - 10.8um)")
fig.colorbar(im2, ax=axes[2], label="Difference (K)")

for ax in axes:
    ax.set_xlabel("Pixel Column (Longitude index)")
    ax.set_ylabel("Pixel Row (Latitude index)")

plt.tight_layout()
plt.savefig("algeria_cropped_preview.png", dpi=200)
print("\nPlot saved successfully as 'algeria_cropped_preview.png'!")
plt.show()