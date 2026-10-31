import os
import shutil
from datetime import datetime
import eumdac

# ==========================================
# 1. YOUR API CREDENTIALS
# Paste your keys inside the quotes below:
# ==========================================
CONSUMER_KEY = "EhZhSU9MOW6R_hZAWre2FfXwBzka"
CONSUMER_SECRET = "bxkGaNxkn7ocgn3eSjxurOwPdOoa"

# ==========================================
# 2. TARGET TIME: July 21, 2026 at 12:10 UTC
# ==========================================
# Rapid Scan files are produced every 5 minutes.
# Setting a 5-minute window returns exactly 1 file.
start_time = datetime(2026, 7, 21, 12, 10)
end_time = datetime(2026, 7, 21, 12, 15)

# Rapid Scan SEVIRI Level 1.5 Collection ID
COLLECTION_ID = "EO:EUM:DAT:MSG:MSG15-RSS"

print("Authenticating with EUMETSAT servers...")
credentials = (CONSUMER_KEY, CONSUMER_SECRET)
token = eumdac.AccessToken(credentials)
print(f"Authentication successful! Token expires at: {token.expiration}")

# Connect to the Data Store
datastore = eumdac.DataStore(token)
selected_collection = datastore.get_collection(COLLECTION_ID)
print(f"Connected to collection: {selected_collection.title}")

# Search for the specific 5-minute scan
print(f"\nSearching for scan between {start_time} and {end_time} UTC...")
products = selected_collection.search(dtstart=start_time, dtend=end_time)

# Download the retrieved product
downloaded = False
for product in products:
    print(f"\nFound product: {product}")
    print("Initiating file stream to disk (approx. 100 MB)...")

    # Stream the file from the server directly to your local drive
    with product.open() as remote_file:
        local_filename = remote_file.name
        with open(local_filename, mode="wb") as local_file:
            shutil.copyfileobj(remote_file, local_file)

    print(f"Download complete: {local_filename}")
    downloaded = True

if not downloaded:
    print("\nNo product found for that exact time window. Double-check the dates.")