# ml/train.py
"""
Streaming LightGBM trainer (file-by-file) for two targets:
 - failure_next_day
 - failure_next_7_days

Key features (Version B - Robust):
 - Processes ONE parquet file at a time (low RAM).
 - Uses a single master feature schema (ml/artifacts/feature_columns.json).
 - STRICT alignment: missing features filled with 0, extra features ignored.
 - GLOBAL_SCALE_POS_WEIGHT from hyperparameters.
 - LR Floor: Prevents learning rate from vanishing.
 - Checkpointing: Saves models every 10 files to prevent total data loss on crash.
 - Empty Batch Skip: Skips training on files with 0 failures to improve stability.
"""

import os
import argparse
import json
import shutil
import gc
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# local utils
from ml.utils import hyperparams as hp

# preprocessing helpers (must exist and be stable)
from ml.data_processing.feature_engineering import load_saved_schema, engineer_features

# paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# data folder: C:/hdd_prediction_ML/data/
DATA_ROOT = os.path.join(REPO_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

# artifacts: C:/hdd_prediction_ML/ml/artifacts/
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "ml", "artifacts")
FEATURE_SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# model dir: C:/hdd_prediction_ML/ml/model/
MODEL_DIR = os.path.join(REPO_ROOT, "ml", "model")
VERSIONS_DIR = os.path.join(MODEL_DIR, "versions")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VERSIONS_DIR, exist_ok=True)

MODEL_PATH_DAY = os.path.join(MODEL_DIR, "model_failure_next_day.txt")
MODEL_PATH_7D = os.path.join(MODEL_DIR, "model_failure_next_7_days.txt")

LABEL_DAY = "failure_next_day"
LABEL_7D = "failure_next_7_days"

# columns to always exclude from feature set (leakage prevention)
EXCLUDE_COLS = { "device_id", "date", "failure", LABEL_DAY, LABEL_7D }

# -------------------------
# Helpers
# -------------------------
def list_processed_parquets() -> List[str]:
    if not os.path.exists(PROCESSED_DIR):
        return []
    files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith(".parquet")]
    return sorted(files)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def load_master_feature_list() -> List[str]:
    """
    Reads the master feature schema saved by feature_engineering.
    """
    if not os.path.exists(FEATURE_SCHEMA_PATH):
        raise FileNotFoundError(f"Feature schema not found at {FEATURE_SCHEMA_PATH}")
    with open(FEATURE_SCHEMA_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "feature_columns" in data:
            return list(data["feature_columns"])
        if "features" in data:
            return list(data["features"])
    if isinstance(data, list):
        return data
    raise RuntimeError("Unrecognized feature schema format in feature_columns.json")

def align_and_order_features(df: pd.DataFrame, master_features: List[str]) -> pd.DataFrame:
    """
    Ensure df contains exactly the master_features in the same order.
    - missing features -> column with zeros
    - extra features -> ignored
    """
    out = df.copy()
    # 1. Add missing cols
    missing = [f for f in master_features if f not in out.columns]
    if missing:
        # Create a DataFrame of zeros for missing columns
        zeros = pd.DataFrame(0, index=out.index, columns=missing)
        out = pd.concat([out, zeros], axis=1)

    # 2. Select strictly master features
    out = out[master_features]
    
    # 3. Optimize types
    for c in out.columns:
        if not np.issubdtype(out[c].dtype, np.number):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    
    out = out.astype(np.float32)
    return out

def load_booster(path: str) -> Optional[lgb.Booster]:
    if not os.path.exists(path):
        return None
    try:
        return lgb.Booster(model_file=path)
    except Exception:
        return None

def save_booster(booster: lgb.Booster, path: str):
    booster.save_model(path)

def archive_existing_models():
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for m in (MODEL_PATH_DAY, MODEL_PATH_7D):
        if os.path.exists(m):
            dst = os.path.join(VERSIONS_DIR, f"{ts}_{os.path.basename(m)}")
            shutil.move(m, dst)

# -------------------------
# Training logic
# -------------------------
def train_stream(files: List[str], mode: str, rounds_per_file: int, shuffle_initial: bool=False):
    if not files:
        raise SystemExit("No processed files found to train on.")

    master_features = load_master_feature_list()
    print(f"Master feature count: {len(master_features)}")

    params_base = dict(hp.LGB_PARAMS)
    base_lr = float(params_base.get("learning_rate", 0.01))
    lr_decay = float(getattr(hp, "LR_DECAY_PER_FILE", 1.0))
    global_spw = getattr(hp, "GLOBAL_SCALE_POS_WEIGHT", None)
    
    # [Improvement] Minimum Learning Rate Floor
    MIN_LR = 0.0001

    if global_spw is None:
        raise SystemExit("GLOBAL_SCALE_POS_WEIGHT must be set in ml/utils/hyperparams.py")

    # Set up file lists
    if mode == "initial":
        n_total = len(files)
        split = max(1, int(n_total * 0.9))
        train_files = files[:split]
        val_files = files[split:]
        print(f"[initial] training on {len(train_files)} files, validation on {len(val_files)} files.")
    else:
        train_files = files
        val_files = []
        print(f"[incremental] training on {len(train_files)} files.")

    # Load or Archive models
    if mode == "incremental":
        booster_day = load_booster(MODEL_PATH_DAY)
        booster_7d = load_booster(MODEL_PATH_7D)
        if booster_day is None or booster_7d is None:
            raise SystemExit("Incremental mode requested but existing model(s) not found.")
        archive_existing_models()
    else:
        booster_day = None
        booster_7d = None
        archive_existing_models()

    # --- Streaming Loop ---
    for i, path in enumerate(train_files, start=1):
        filename = os.path.basename(path)
        print(f"\n[{i}/{len(train_files)}] Loading {filename}")
        
        try:
            df = load_parquet(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}; skipping.")
            continue

        df = engineer_features(df, lazy=True)
        X = align_and_order_features(df, master_features)

        # [Improvement] Calculate decayed LR but clamp it to MIN_LR
        current_lr = max(base_lr * (lr_decay ** (i-1)), MIN_LR)

        # --- Train Day Model ---
        if LABEL_DAY in df.columns:
            # [Improvement] Skip if no positive labels (saves time/stability)
            pos_count = df[LABEL_DAY].sum()
            if pos_count > 0:
                y_day = df[LABEL_DAY].astype(int)
                params_day = dict(params_base)
                params_day["scale_pos_weight"] = float(global_spw)
                params_day["learning_rate"] = current_lr
                
                dtrain = lgb.Dataset(X, label=y_day, free_raw_data=False)
                print(f" Training {LABEL_DAY}: rows={len(y_day)} (+={pos_count}) lr={current_lr:.5f}")
                
                booster_day = lgb.train(
                    params_day, dtrain, 
                    num_boost_round=rounds_per_file, 
                    init_model=booster_day, 
                    keep_training_booster=True
                )
            else:
                print(f" Skipping {LABEL_DAY}: No failures in this file.")

        # --- Train 7D Model ---
        if LABEL_7D in df.columns:
            pos_count = df[LABEL_7D].sum()
            if pos_count > 0:
                y_7d = df[LABEL_7D].astype(int)
                params_7d = dict(params_base)
                params_7d["scale_pos_weight"] = float(global_spw)
                params_7d["learning_rate"] = current_lr
                
                dtrain7 = lgb.Dataset(X, label=y_7d, free_raw_data=False)
                print(f" Training {LABEL_7D}: rows={len(y_7d)} (+={pos_count}) lr={current_lr:.5f}")
                
                booster_7d = lgb.train(
                    params_7d, dtrain7, 
                    num_boost_round=rounds_per_file, 
                    init_model=booster_7d, 
                    keep_training_booster=True
                )
            else:
                print(f" Skipping {LABEL_7D}: No failures in this file.")

        # Cleanup memory
        del df, X
        gc.collect()

        # [Improvement] Intermediate Checkpoints (Every 10 files)
        if i % 10 == 0:
            print(f" >> Checkpointing models at step {i}...")
            if booster_day: save_booster(booster_day, MODEL_PATH_DAY)
            if booster_7d: save_booster(booster_7d, MODEL_PATH_7D)

    # --- Final Save ---
    if booster_day:
        save_booster(booster_day, MODEL_PATH_DAY)
        print(f"Final model saved: {MODEL_PATH_DAY}")
    if booster_7d:
        save_booster(booster_7d, MODEL_PATH_7D)
        print(f"Final model saved: {MODEL_PATH_7D}")

    # --- Validation (Initial Mode Only) ---
    if mode == "initial" and val_files:
        print("\nRunning validation on reserved files...")
        y_true_day, y_pred_day = [], []
        y_true_7d, y_pred_7d = [], []

        for path in val_files:
            try:
                dfv = load_parquet(path)
                dfv = engineer_features(dfv, lazy=True)
                Xv = align_and_order_features(dfv, master_features)
                
                if LABEL_DAY in dfv.columns and booster_day:
                    if dfv[LABEL_DAY].nunique() >= 2:
                        y_true_day.append(dfv[LABEL_DAY].astype(int).values)
                        y_pred_day.append(booster_day.predict(Xv))
                
                if LABEL_7D in dfv.columns and booster_7d:
                    if dfv[LABEL_7D].nunique() >= 2:
                        y_true_7d.append(dfv[LABEL_7D].astype(int).values)
                        y_pred_7d.append(booster_7d.predict(Xv))
                
                del dfv, Xv
                gc.collect()
            except Exception as e:
                print(f"Val error on {path}: {e}")

        # Compute Metrics
        if y_true_day:
            yt = np.concatenate(y_true_day)
            yp = np.concatenate(y_pred_day)
            if len(np.unique(yt)) > 1:
                print(f"[VALIDATION][{LABEL_DAY}] AUC = {roc_auc_score(yt, yp):.4f}")
        
        if y_true_7d:
            yt = np.concatenate(y_true_7d)
            yp = np.concatenate(y_pred_7d)
            if len(np.unique(yt)) > 1:
                print(f"[VALIDATION][{LABEL_7D}] AUC = {roc_auc_score(yt, yp):.4f}")

    print("Training finished.")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog="ml.train")
    parser.add_argument("--mode", choices=["initial", "incremental"], default="initial")
    parser.add_argument("--n_files", type=int, default=None, help="Num files to process")
    parser.add_argument("--rounds", type=int, default=None, help="Boost rounds per file")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle initial training files")
    args = parser.parse_args()

    files = list_processed_parquets()
    if not files:
        raise SystemExit("No processed parquet files found in data/processed")

    if args.mode == "initial":
        n = args.n_files or 200
        if args.shuffle:
            rng = np.random.default_rng(42)
            nsel = min(n, len(files))
            sel = list(rng.choice(files, size=nsel, replace=False))
            sel = sorted(sel)
        else:
            sel = files[:min(n, len(files))]
    else:
        n = args.n_files or 7
        sel = files[-min(n, len(files)):]

    rounds = args.rounds or getattr(hp, "DEFAULT_ROUNDS_PER_FILE", 30)
    train_stream(sel, args.mode, rounds, shuffle_initial=args.shuffle)

if __name__ == "__main__":
    main()