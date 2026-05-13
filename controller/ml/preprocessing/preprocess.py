import os, json, pickle, gc
from typing import Dict, List
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from controller.ml.preprocessing.clean import basic_clean, save_parquet
from controller.ml.preprocessing.feature_engineering import compute_features_for_file
from controller.ml.preprocessing.add_labels import build_label_map_from_timelines
from controller.config import PROCESSED_CSV_DIR, CHECKPOINT_DIR

# --- DYNAMIC CHECKPOINT PATHS ---
DONE_F = CHECKPOINT_DIR / "preprocessing" / "processed_raws.json"

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return default

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def build_timelines_with_checkpoint(raw_files: List[str]) -> Dict:
    """Phase 1: High-speed metadata scan [cite: 329-330]."""
    timelines = defaultdict(list)
    for path in tqdm(raw_files, desc="[1/3] Scanning Timelines"):
        df = pd.read_csv(path, usecols=lambda c: c in ["device_id", "serial_number", "date", "failure"])
        df = basic_clean(df)
        df = df.dropna(subset=["date"])
        for dev, grp in df.groupby("device_id", sort=False):
            records = [{"date": d, "failure": f} for d, f in zip(grp["date"], grp["failure"])]
            timelines[str(dev)].extend(records)
        del df; gc.collect()
    return timelines

def process_raws_with_checkpoint(raw_files: List[str]):
    """Phase 2: Feature Engineering with guaranteed Parquet output [cite: 331-333]."""
    done = set(load_json(DONE_F, []))
    
    for path in tqdm(raw_files, desc="[2/3] Engineering Features"):
        # We still skip the heavy math if done...
        if path in done:
            # BUT: We must ensure the Parquet is actually there!
            # If a Parquet is missing, we re-run this file.
            log_name = Path(path).stem
            if any(PROCESSED_CSV_DIR.glob(f"*{log_name}*.parquet")):
                continue

        df = basic_clean(pd.read_csv(path))
        if df.empty: 
            done.add(path)
            continue
        
        feat = compute_features_for_file(df)
        
        final = pd.concat([
            df[["device_id", "date", "failure"]].reset_index(drop=True), 
            feat.reset_index(drop=True)
        ], axis=1) 
        
        # Ensure Parquets are written BEFORE we mark as done
        for date_str, g in final.groupby(final["date"].dt.strftime("%Y-%m-%d")):
            save_parquet(g, PROCESSED_CSV_DIR / f"{date_str}.parquet")
            
        done.add(path)
        save_json(DONE_F, sorted(list(done)))
        del df, feat, final; gc.collect()

def finalize_labels(date_label_map: Dict):
    """Phase 3: Label Enrichment [cite: 334-337]."""
    for date_str, devmap in tqdm(date_label_map.items(), desc="[3/3] Finalizing Labels"):
        path = PROCESSED_CSV_DIR / f"{date_str}.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        labels_df = pd.DataFrame([{"device_id": d, **v} for d, v in devmap.items()])
        df = df.merge(labels_df, on="device_id", how="left", suffixes=("", "_new"))
        for col in ["failure_next_day", "failure_next_7_days"]:
            if col + "_new" in df.columns:
                df[col] = df[col + "_new"].fillna(0).astype(int)
        df = df.drop(columns=[c for c in df.columns if "_new" in c])
        df.to_parquet(path, index=False)
        del df, labels_df; gc.collect()