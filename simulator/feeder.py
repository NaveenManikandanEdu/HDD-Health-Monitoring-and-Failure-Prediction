import shutil
import time
import os
from pathlib import Path
from datetime import datetime

# ==============================
# ABSOLUTE ANCHORING
# ==============================
SIM_DIR = Path(__file__).resolve().parent
BASE_DIR = SIM_DIR.parent
SOURCE_RAW = SIM_DIR / "raw_snapshots"
TARGET_ARCHIVE = BASE_DIR / "controller" / "storage" / "archive"

# ==============================
# CONFIGURATION
# ==============================
# Changed to "production" for the 5-minute requirement
MODE = "production" 

INTERVALS = {
    "test": 10,        # 10 seconds for rapid testing
    "production": 300   # 300 seconds = 5 Minutes [Your Request]
}
INTERVAL = INTERVALS.get(MODE, 300)

# ==============================

def run_feeder():
    print("=" * 60)
    print(" [SIMULATOR] Data Feeder Active")
    print(f" Mode:           {MODE.upper()}")
    print(f" Frequency:      Every {INTERVAL/60:.1f} minutes")
    print(f" Target:         {TARGET_ARCHIVE}")
    print("=" * 60)

    # Ensure directories exist 
    SOURCE_RAW.mkdir(parents=True, exist_ok=True)
    TARGET_ARCHIVE.mkdir(parents=True, exist_ok=True)

    while True:
        # Get oldest snapshot first
        snapshots = sorted(list(SOURCE_RAW.glob("*.csv")))

        if snapshots:
            file_to_move = snapshots[0]
            dest_path = TARGET_ARCHIVE / file_to_move.name

            try:
                # Absolute path move to prevent WinError race conditions
                shutil.move(str(file_to_move), str(dest_path))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent → {file_to_move.name}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Move Failed: {e}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Staging empty. Waiting for snapshots...")

        # Sleep for the 5-minute interval
        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        run_feeder()
    except KeyboardInterrupt:
        print("\n[!] Simulator feeder shut down by user.")