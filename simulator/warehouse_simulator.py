import time
import json
import pandas as pd
import requests
import os
import sys
import random

# --- DYNAMIC PATH FIX ---
# This ensures that even if you run from the root, the 'simulator' folder is added to Python's search path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Now the import will work perfectly
try:
    from warehouse_config import WAREHOUSES, CSV_PATH, BACKEND_URL, CHUNK_SIZE, SEND_INTERVAL
except ImportError:
    print("❌ Fatal Error: Could not find warehouse_config.py in the simulator directory.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGET_FAILURE_RATE = 0.05 

def sanitize_chunk(chunk):
    sanitized = []
    for _, row in chunk.iterrows():
        record = row.to_dict()
        # 95% of the time, force the drive to look perfectly healthy
        if random.random() > TARGET_FAILURE_RATE:
            record["smart_5_raw"] = 0     
            record["smart_187_raw"] = 0  
            record["smart_197_raw"] = 0  
            record["smart_198_raw"] = 0  
            record["smart_194_raw"] = random.randint(30, 45) # Normal temp
        sanitized.append(record)
    return sanitized

def main():
    print(f"🚀 Live Stream Simulator Started")
    
    if CSV_PATH is None:
        print(f"❌ Error: No CSV found in simulator/data/")
        return

    print(f"📂 Reading Simulation Data from: {CSV_PATH}")

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Normalize columns
    df.columns = df.columns.str.strip()
    if "device_id" not in df.columns and "serial_number" in df.columns:
        df["device_id"] = df["serial_number"]

    print(f"✅ Loaded {len(df)} rows.")
    print(f"🛡️  Safety Mode: 95% of drives forced Healthy")
    print("📡 Simulator Active. Pushing telemetry to API...")

    while True:
        for wh_id, vaults in WAREHOUSES.items():
            for vault_id in vaults:
                # 1. Grab random chunk
                try:
                    chunk = df.sample(n=CHUNK_SIZE)
                except ValueError:
                    chunk = df 

                # 2. Sanitize (Inject failures vs health)
                clean_records = sanitize_chunk(chunk.fillna(0))

                # 3. Payload
                payload = {
                    "warehouse_id": wh_id,
                    "vault_id": vault_id,
                    "timestamp": str(pd.Timestamp.now()),
                    "records": clean_records
                }

                # 4. Send
                try:
                    res = requests.post(f"{BACKEND_URL}/live/update", json=payload, timeout=5)
                    if res.status_code == 200:
                        print(f"✅ Sent {len(clean_records)} records to {wh_id}::{vault_id}")
                    else:
                        print(f"⚠️ Backend Error ({res.status_code}): {res.text}")
                except requests.exceptions.RequestException:
                    print("❌ Connection Failed. Ensure FastAPI backend is running on 127.0.0.1:8000")
        
        time.sleep(SEND_INTERVAL)

if __name__ == "__main__":
    main()