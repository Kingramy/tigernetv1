import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration matching our cropped satellite grid
WIDTH = 250
HEIGHT = 120
LON_MIN, LON_MAX = -2.0, 8.5
LAT_MIN, LAT_MAX = 34.0, 37.5

# 2. Load the NASA FIRMS dataset
excel_file = "nasa data set.xlsx"
print(f"Loading NASA targets from: {excel_file}...")
df = pd.read_excel(excel_file)

# 3. Filter for our specific high-intensity fire date: July 21, 2026 (12:12 UTC)
# acq_time 1212 corresponds to the midday overpass
target_date = "2026-07-21"
target_time = 1212

fire_subset = df[(df["acq_date"] == target_date) & (df["acq_time"] == target_time)]
print(f"Found {len(fire_subset)} fire detections for {target_date} at {target_time} UTC.")

# 4. Initialize an empty binary mask with zeros: [Height, Width]
ground_truth_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# 5. Map each GPS fire coordinate to its corresponding pixel index
fires_mapped = 0
for _, row in fire_subset.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    frp = row["frp"]

    # Verify the coordinate falls inside our Northern Algeria bounding box
    if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
        # Calculate pixel indices using the coordinate equations
        x = int(((lon - LON_MIN) / (LON_MAX - LON_MIN)) * WIDTH)
        y = int(((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * HEIGHT)

        # Ensure calculated indices remain within matrix boundaries
        x = min(max(x, 0), WIDTH - 1)
        y = min(max(y, 0), HEIGHT - 1)

        # Mark the active fire pixel
        ground_truth_mask[y, x] = 1.0
        fires_mapped += 1

print(f"Successfully mapped {fires_mapped} fire pixels onto the (120 x 250) mask.")
print(f"Total non-fire pixels (0.0): {np.sum(ground_truth_mask == 0.0)}")
print(f"Total fire pixels     (1.0): {np.sum(ground_truth_mask == 1.0)}")

# 6. Save the ground-truth mask as a raw NumPy array
np.save("sample_ground_truth_mask.npy", ground_truth_mask)
print("\nSaved mask array to 'sample_ground_truth_mask.npy'!")

# 7. Visualize the binary mask
plt.figure(figsize=(10, 5))
plt.imshow(ground_truth_mask, cmap="hot", origin="upper")
plt.title(f"Ground Truth Binary Mask (Y) - {target_date} {target_time} UTC")
plt.xlabel("Pixel Column (Longitude)")
plt.ylabel("Pixel Row (Latitude)")
plt.colorbar(label="Class (0 = Safe, 1 = Active Fire)")
plt.tight_layout()
plt.savefig("ground_truth_mask_preview.png", dpi=200)
print("Saved preview image to 'ground_truth_mask_preview.png'!")
plt.show()