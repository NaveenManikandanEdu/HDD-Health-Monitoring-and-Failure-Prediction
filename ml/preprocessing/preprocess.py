# ml/preprocessing/preprocess.py
from __future__ import annotations

import os
import glob
import json
import pickle
from typing import Dict, List
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from ml.preprocessing.clean import basic_clean, save_parquet
from ml.preprocessing.feature_engineering import compute_features_for_file
from ml.preprocessing.add_labels import build_label_map_from_timelines

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

CKPT_DIR = "ml/preprocessing/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

STATE_F = os.path.join(CKPT_DIR, "state.json")
TIMELINE_F = os.path.join(CKPT_DIR, "timelines.pkl")
DONE_F = os.path.join(CKPT_DIR, "processed_raws.json")

# ------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_pickle(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ------------------------------------------------------------
# Phase 1 — Build device timelines (RESUMABLE)
# ------------------------------------------------------------
def build_timelines_with_checkpoint(raw_files: List[str]) -> Dict:
    print("\n[1/3] Building device timelines (RESUMABLE, low RAM)")

    timelines = (
        load_pickle(TIMELINE_F)
        if os.path.exists(TIMELINE_F)
        else defaultdict(list)
    )

    state = load_json(STATE_F, {})
    start_idx = state.get("timeline_index", 0)

    for i in tqdm(
        range(start_idx, len(raw_files)),
        total=len(raw_files),
        initial=start_idx,
        desc="Scanning raw CSVs",
        unit="file",
        smoothing=0,
        dynamic_ncols=True,
    ):
        path = raw_files[i]

        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in ["device_id", "serial_number", "date", "failure"],
            )
        except Exception:
            continue

        if df.empty:
            continue

        if "device_id" not in df.columns:
            df["device_id"] = df["serial_number"].astype(str)
        else:
            df["device_id"] = df["device_id"].astype(str)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["failure"] = (
            pd.to_numeric(df.get("failure", 0), errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
        )

        for dev, grp in df.groupby("device_id", sort=False):
            timelines[dev].extend(
                grp[["date", "failure"]]
                .dropna()
                .to_dict("records")
            )

        # ---- CHECKPOINT ----
        save_pickle(TIMELINE_F, timelines)
        save_json(STATE_F, {"timeline_index": i + 1})

    print("✓ Timeline building complete")
    return timelines

# ------------------------------------------------------------
# Phase 2 — Feature engineering + parquet writing (RESUMABLE)
# ------------------------------------------------------------
def process_raws_with_checkpoint(raw_files: List[str]):
    print("\n[2/3] Feature engineering + parquet writing (RESUMABLE)")

    done = set(load_json(DONE_F, []))

    for path in tqdm(
        raw_files,
        desc="Processing raw CSVs",
        unit="file",
        dynamic_ncols=True,
    ):
        if path in done:
            continue

        df_raw = pd.read_csv(path)
        df = basic_clean(df_raw)

        if df.empty:
            done.add(path)
            save_json(DONE_F, sorted(done))
            continue

        # Feature engineering
        feat = compute_features_for_file(df)

        base = pd.DataFrame({
            "device_id": df["device_id"].astype(str),
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "failure": df.get("failure", 0).astype(int),
        })

        final = pd.concat(
            [base.reset_index(drop=True), feat.reset_index(drop=True)],
            axis=1,
        )

        final["date_str"] = final["date"].dt.strftime("%Y-%m-%d")

        # ---- Nested progress: per-day parquet writing ----
        for date_str, g in tqdm(
            final.groupby("date_str"),
            desc="  Writing per-day parquet",
            unit="day",
            leave=False,
            dynamic_ncols=True,
        ):
            out = g.drop(columns="date_str")
            out["failure_next_day"] = 0
            out["failure_next_7_days"] = 0
            save_parquet(out, os.path.join(OUT_DIR, f"{date_str}.parquet"))

        # ---- CHECKPOINT ----
        done.add(path)
        save_json(DONE_F, sorted(done))

    print("✓ Feature engineering complete")

# ------------------------------------------------------------
# Phase 3 — Finalize labels (SAFE TO RERUN)
# ------------------------------------------------------------
def finalize_labels(date_label_map: Dict):
    print("\n[3/3] Finalizing labels")

    for date_str, devmap in tqdm(
        date_label_map.items(),
        total=len(date_label_map),
        desc="Finalizing labels",
        unit="date",
        smoothing=0,
        dynamic_ncols=True,
    ):
        path = os.path.join(OUT_DIR, f"{date_str}.parquet")
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)

        labels_df = pd.DataFrame([
            {
                "device_id": dev,
                "failure_next_day": v["failure_next_day"],
                "failure_next_7_days": v["failure_next_7_days"],
            }
            for dev, v in devmap.items()
        ])

        if labels_df.empty:
            continue

        df = df.merge(labels_df, on="device_id", how="left", suffixes=("", "_new"))

        for col in ["failure_next_day", "failure_next_7_days"]:
            if col + "_new" in df.columns:
                df[col] = df[col + "_new"].fillna(0).astype(int)
                df.drop(columns=col + "_new", inplace=True)

        df.to_parquet(path, index=False)

    print("✓ Labels finalized")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"Found {len(raw_files)} raw CSV files")

    timelines = build_timelines_with_checkpoint(raw_files)
    date_label_map = build_label_map_from_timelines(timelines)

    process_raws_with_checkpoint(raw_files)
    finalize_labels(date_label_map)

    print("\n✅ PREPROCESSING COMPLETE (RESUMABLE SAFE MODE)")

if __name__ == "__main__":
    main()
