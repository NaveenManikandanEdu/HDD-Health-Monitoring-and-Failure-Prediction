import os
import glob

# --- PATH CONFIGURATION ---
# Get the directory where this config file is located (simulator/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the data folder
DATA_DIR = os.path.join(BASE_DIR, "data")

# Automatically find ANY CSV file inside the data folder
found_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if found_files:
    # Take the first CSV found (e.g., "2025-09-29.csv" or "dataset_v2.csv")
    CSV_PATH = found_files[0]
else:
    CSV_PATH = None # Will be handled by the scripts
    print(f"⚠️  WARNING: No CSV file found in {DATA_DIR}")

# Warehouse Definitions
WAREHOUSES = {
    "chennai": ["Vault_1028", "Vault_1029"],
    "bangalore": ["Vault_2055", "Vault_2056"]
}

BACKEND_URL = "http://127.0.0.1:8000"
CHUNK_SIZE = 50   # Number of rows to send per request
SEND_INTERVAL = 2 # Seconds between requests
