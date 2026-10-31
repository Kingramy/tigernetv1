import os
import glob
from satpy import Scene

# 1. Locate the .nat file
raw_data_path = "MSG4-SEVI-MSG15-0100-NA-20260721121418.623000000Z-NA.nat"

if os.path.isfile(raw_data_path):
    target_file = raw_data_path
else:
    nat_files = glob.glob(os.path.join(raw_data_path, "*.nat")) or glob.glob("*.nat")
    if not nat_files:
        raise FileNotFoundError(f"No .nat files found at {raw_data_path} or current directory!")
    target_file = nat_files[0]

print(f"Loading satellite file: {target_file}\n")

# 2. Initialize the Satpy Scene with the EUMETSAT Native Reader
scn = Scene(filenames=[target_file], reader="seviri_l1b_native")

# 3. Inspect all available bands stored inside the file
print("--- Available Datasets Inside This File ---")
available_channels = scn.available_dataset_names()
print(available_channels)
print(f"\nTotal channels found: {len(available_channels)}")

# 4. Load the two primary thermal channels needed for wildfire detection
print("\nLoading IR_039 (3.9 um) and IR_108 (10.8 um)...")
scn.load(["IR_039", "IR_108"])

# 5. Inspect the loaded bands
ir_39 = scn["IR_039"]
ir_108 = scn["IR_108"]

print("\n--- Channel Details ---")
print(f"IR_039 Name: {ir_39.attrs.get('name')}")
print(f"IR_039 Units: {ir_39.attrs.get('units')} (Brightness Temperature)")
print(f"IR_039 Shape (Height x Width): {ir_39.shape}")

print(f"\nIR_108 Name: {ir_108.attrs.get('name')}")
print(f"IR_108 Units: {ir_108.attrs.get('units')} (Brightness Temperature)")
print(f"IR_108 Shape (Height x Width): {ir_108.shape}")

import numpy as np

# 6. Check a small slice of actual temperature values in Kelvin
data_sample = ir_39.values
print(f"\nMinimum value in IR_039 (ignoring NaNs): {np.nanmin(data_sample):.2f} K")
print(f"Maximum value in IR_039 (ignoring NaNs): {np.nanmax(data_sample):.2f} K")