"""
ml/data_processing/fix_labels.py

Post-Hoc Label Generator ("Time Travel" Fix)
--------------------------------------------
Problem: Streaming pipelines process data day-by-day and cannot look into the future
to generate "Next Day Failure" labels.

Solution:
1. Scan ALL files to build a master index of {device_id: [failure_dates]}.
2. Re-process every file, comparing current dates against the failure index.
3. Correctly backfill 'failure_next_day' and 'failure_next_7_days'.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta

# Auto-detect path
def find_project_root():
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "data", "processed")): return cwd
    curr = os.path.dirname(os.path.abspath(__file__))
    while len(curr) > 4:
        if os.path.exists(os.path.join(curr, "data", "processed")): return curr
        curr = os.path.dirname(curr)
    return None

BASE_DIR = find_project_root()
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def fix_labels():
    print(f"🔧 Starting Label Correction in: {PROCESSED_DIR}")
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    
    if not files:
        print("❌ No files found.")
        return

    # =========================================================
    # PASS 1: Build the Failure Map (The "Future" Knowledge)
    # =========================================================
    print("🔍 PASS 1: Scanning for failure events...")
    failure_records = []

    for fpath in files:
        try:
            # Read only essential cols
            df = pd.read_parquet(fpath, columns=["device_id", "date", "failure"])
            # Filter for failures
            failures = df[df["failure"] == 1].copy()
            if not failures.empty:
                failure_records.append(failures)
        except Exception:
            pass

    if not failure_records:
        print("❌ No failures found in dataset. Cannot generate labels.")
        return

    # Combine into a single lookup table
    all_failures = pd.concat(failure_records)
    all_failures["failure_date"] = pd.to_datetime(all_failures["date"])
    
    # Simplify: Keep only Device ID and Failure Date
    # Rename to avoid collision during merge
    failure_map = all_failures[["device_id", "failure_date"]].rename(
        columns={"failure_date": "target_date"}
    )

    print(f"📘 Found {len(failure_map)} failure events. Mapping complete.")

    # =========================================================
    # PASS 2: Apply Labels to History
    # =========================================================
    print(f"📝 PASS 2: Backfilling labels across {len(files)} files...")

    fixed_count = 0

    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            
            # Convert date to datetime if not already
            df["date"] = pd.to_datetime(df["date"])
            
            # Drop old incorrect labels if they exist
            cols_to_drop = [c for c in ["failure_next_day", "failure_next_7_days"] if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # 1. Merge with failure map
            # This attaches the *future* failure date to the *current* row if IDs match
            # Note: A device might appear multiple times in failure_map, 
            # so we merge and calculate distance, then filter.
            
            # Inner join? No, Left join. Most devices never fail.
            merged = pd.merge(df, failure_map, on="device_id", how="left")

            # 2. Calculate Time Delta (Future Failure - Current Date)
            merged["days_to_failure"] = (merged["target_date"] - merged["date"]).dt.days

            # 3. Generate Target: Next Day (1 day before failure)
            # Logic: If failure is tomorrow (diff == 1)
            merged["failure_next_day"] = (merged["days_to_failure"] == 1).astype("float32")

            # 4. Generate Target: Next 7 Days (1 to 7 days before failure)
            # Logic: If failure is within 7 days in the future
            # Note: We usually exclude day 0 (the failure itself) from "prediction", 
            # or include it. Let's include 1 to 7.
            merged["failure_next_7_days"] = (
                (merged["days_to_failure"] >= 1) & 
                (merged["days_to_failure"] <= 7)
            ).astype("float32")

            # Handle duplicates:
            # If a device failed twice (rare), merge creates rows. 
            # We want to keep the original rows and just take the 'max' label (1 if ANY failure is close).
            # But 'merge' explodes the dataframe size if multiple failures match.
            # Efficient Fix: Group by index and take max.
            
            if len(merged) > len(df):
                # Collapse back to original shape
                targets = merged.groupby(merged.index)[["failure_next_day", "failure_next_7_days"]].max()
                # Join back to original df
                df["failure_next_day"] = targets["failure_next_day"]
                df["failure_next_7_days"] = targets["failure_next_7_days"]
            else:
                # Simple case
                df["failure_next_day"] = merged["failure_next_day"]
                df["failure_next_7_days"] = merged["failure_next_7_days"]

            # Fill NaNs (healthy drives have no match in failure_map)
            df["failure_next_day"] = df["failure_next_day"].fillna(0.0)
            df["failure_next_7_days"] = df["failure_next_7_days"].fillna(0.0)

            # Save back
            df.to_parquet(fpath, index=False)
            
            fixed_count += 1
            if fixed_count % 10 == 0:
                print(f"   Processed {fixed_count}/{len(files)} files...")

        except Exception as e:
            print(f"❌ Error fixing {os.path.basename(fpath)}: {e}")

    print("\n✅ Label Correction Complete.")
    print("   Run 'check_failure_stats.py' again to verify positive samples.")

if __name__ == "__main__":
    fix_labels()