
import os
import shutil
import time
import zipfile
from datetime import datetime, timedelta
import eumdac
import pandas as pd

# =====================================================================
# 1. CONFIGURATION & CREDENTIALS
# =====================================================================

CONSUMER_KEY = "EhZhSU9MOW6R_hZAWre2FfXwBzka"
CONSUMER_SECRET = "bxkGaNxkn7ocgn3eSjxurOwPdOoa"

EXCEL_FILE = "nasa data set.xlsx"
OUTPUT_DIR = r"D:\FINDFIRE_DATA"
COLLECTION_ID = "EO:EUM:DAT:MSG:MSG15-RSS"
TARGET_DOWNLOAD_COUNT = 400

# Set to True to delete the .zip archive after extracting the .nat file (saves ~40 GB)
DELETE_ZIP_AFTER_EXTRACT = True

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2. SELECT THE TOP 400 FIRE SCENES FROM NASA DATA
# =====================================================================
print(f"Loading NASA detections from '{EXCEL_FILE}'...")
df = pd.read_excel(EXCEL_FILE)

# Group detections occurring at the exact same date & time
scenes = (
    df.groupby(["acq_date", "acq_time"])
    .agg(
        fire_count=("latitude", "count"),
        max_frp=("frp", "max"),
        total_frp=("frp", "sum"),
    )
    .reset_index()
)

# Sort by Total Fire Radiative Power (highest energy first)
scenes_sorted = scenes.sort_values(by="total_frp", ascending=False)
top_scenes = scenes_sorted.head(TARGET_DOWNLOAD_COUNT).copy()

print(f"Extracted the top {len(top_scenes)} highest-intensity wildfire scenes.")
print(
    f"Covering {top_scenes['fire_count'].sum():,} fire points (Total FRP range:"
    f" {top_scenes['total_frp'].min():.1f} MW to"
    f" {top_scenes['total_frp'].max():.1f} MW).\n"
)


# =====================================================================
# 3. HELPER FUNCTIONS FOR DATES AND TOKENS
# =====================================================================
def get_auth_token(key, secret):
  """Requests a new AccessToken and connects to the collection."""
  creds = (key, secret)
  token = eumdac.AccessToken(creds)
  datastore = eumdac.DataStore(token)
  collection = datastore.get_collection(COLLECTION_ID)
  return token, collection, time.time()


def parse_timestamp_slot(acq_date, acq_time):
  """Converts NASA acq_date and acq_time into a 5-minute EUMETSAT time window."""
  if isinstance(acq_date, pd.Timestamp):
    d_str = acq_date.strftime("%Y-%m-%d")
  else:
    d_str = str(acq_date)[:10]

  time_str = f"{int(acq_time):04d}"
  hour = int(time_str[:2])
  minute = int(time_str[2:])

  exact_dt = datetime.strptime(
      f"{d_str} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M"
  )

  # Round down to the nearest 5-minute interval (:00, :05, :10, :15, etc.)
  base_minute = (minute // 5) * 5
  dt_start = exact_dt.replace(minute=base_minute, second=0)
  dt_end = dt_start + timedelta(minutes=5)
  return dt_start, dt_end


def extract_archive(zip_filepath, extract_to, remove_zip=True):
  """Unzips the downloaded file and cleans up the zip archive."""
  with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
    zip_ref.extractall(extract_to)
  if remove_zip and os.path.exists(zip_filepath):
    os.remove(zip_filepath)


# =====================================================================
# 4. DOWNLOAD & UNZIP LOOP
# =====================================================================
print("Connecting to EUMETSAT servers...")
token, collection, last_token_refresh = get_auth_token(
    CONSUMER_KEY, CONSUMER_SECRET
)
print("Connected successfully!\n")

success_count = 0
skipped_count = 0
failed_count = 0

for idx, (_, row) in enumerate(top_scenes.iterrows(), start=1):
  # Refresh token every 50 minutes (3000 seconds) to avoid token expiry
  if time.time() - last_token_refresh > 3000:
    print("\n--- Refreshing EUMETSAT API Token (60-minute limit) ---")
    token, collection, last_token_refresh = get_auth_token(
        CONSUMER_KEY, CONSUMER_SECRET
    )

  dt_start, dt_end = parse_timestamp_slot(row["acq_date"], row["acq_time"])
  scene_label = f"{dt_start.strftime('%Y-%m-%d %H:%M')} (FRP: {row['total_frp']:.1f} MW, Points: {row['fire_count']})"

  print(f"[{idx}/{TARGET_DOWNLOAD_COUNT}] Target: {scene_label}")

  try:
    # 1. Query the 5-minute product
    products = list(collection.search(dtstart=dt_start, dtend=dt_end))

    # Expand search window by +/- 5 minutes if satellite clock drifted slightly
    if not products:
      products = list(
          collection.search(
              dtstart=dt_start - timedelta(minutes=5),
              dtend=dt_end + timedelta(minutes=5),
          )
      )

    if not products:
      print(f"   [!] No satellite product found for {dt_start}. Skipping.")
      failed_count += 1
      continue

    product = products[0]
    expected_zip = os.path.join(OUTPUT_DIR, f"{product}.zip")
    expected_nat = os.path.join(OUTPUT_DIR, f"{product}.nat")

    # 2. Check if .nat is already unzipped and ready
    if os.path.exists(expected_nat):
      print(f"   [✓] Already unzipped & ready: {product}.nat (Skipping)")
      skipped_count += 1
      continue

    # 3. Check if .zip exists from a previous run but was not yet unzipped
    if os.path.exists(expected_zip):
      print(f"   [!] Found existing archive. Unzipping {product}.zip...")
      extract_archive(
          expected_zip, OUTPUT_DIR, remove_zip=DELETE_ZIP_AFTER_EXTRACT
      )
      print("   [✓] Extraction complete.")
      success_count += 1
      continue

    # 4. Download the file stream from EUMETSAT
    print(f"   --> Downloading: {product}.zip...")
    start_dl = time.time()
    with product.open() as remote_file:
      with open(expected_zip, mode="wb") as local_file:
        shutil.copyfileobj(remote_file, local_file)

    dl_duration = time.time() - start_dl
    file_size_mb = os.path.getsize(expected_zip) / (1024 * 1024)
    print(f"   --> Downloaded ({file_size_mb:.1f} MB in {dl_duration:.1f}s).")

    # 5. Unzip immediately before moving to the next item
    print("   --> Unzipping archive...")
    extract_archive(
        expected_zip, OUTPUT_DIR, remove_zip=DELETE_ZIP_AFTER_EXTRACT
    )
    print("   [✓] Extracted .nat file ready on disk.")

    success_count += 1

  except Exception as e:
    print(f"   [X] Error processing {dt_start}: {e}")
    failed_count += 1

print("\n==========================================")
print("BATCH DOWNLOAD & EXTRACTION COMPLETE!")
print(f"Total Scenes Target : {TARGET_DOWNLOAD_COUNT}")
print(f"Successfully Ready  : {success_count}")
print(f"Already Existed     : {skipped_count}")
print(f"Failed / Missing    : {failed_count}")
print(f"Output Directory    : {OUTPUT_DIR}")
print("==========================================")