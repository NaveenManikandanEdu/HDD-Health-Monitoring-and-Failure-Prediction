"""
ml/pipeline_stream.py

Streaming preprocessing pipeline (file-by-file):
 - Reads CSV from data/raw/
 - Runs: basic_clean -> engineer_features -> add_labels -> validate
 - Saves parquet to data/processed/
 - Uses small-memory friendly operations and gc.collect()
"""

import os
import random
import argparse
import gc
from tqdm import tqdm

import pandas as pd

from ml.data_processing.clean import basic_clean
from ml.data_processing.feature_engineering import engineer_features
from ml.data_processing.add_labels import add_labels
from ml.data_processing.validate import validate_dataframe

# Directories (project-root / data / raw|processed)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def process_single_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = basic_clean(df)
    df = engineer_features(df, lazy=True)
    df = add_labels(df)
    ok, msg = validate_dataframe(df)
    if not ok:
        raise ValueError(f"Validation failed for {csv_path}: {msg}")
    return df


def run_streaming_pipeline(n_files: int = None, sample_random: bool = False):
    print("STARTING STREAMING PIPELINE (one file at a time)")
    print(f"RAW_DIR: {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")])
    if not all_files:
        print("No CSV files found in data/raw.")
        return

    if n_files:
        if sample_random:
            all_files = random.sample(all_files, min(n_files, len(all_files)))
        else:
            all_files = all_files[:n_files]

    print(f"Total CSVs to process: {len(all_files)}")

    for filename in tqdm(all_files, desc="Processing CSVs", unit="file"):
        csv_path = os.path.join(RAW_DIR, filename)
        tqdm.write(f"Processing file: {filename}")

        try:
            df = process_single_csv(csv_path)
        except Exception as e:
            tqdm.write(f"ERROR processing {filename}: {e}")
            # continue to next file (do not abort full run)
            continue

        # Convert object columns to strings to avoid parquet issues
        object_cols = df.select_dtypes(include=["object"]).columns
        if len(object_cols) > 0:
            df[object_cols] = df[object_cols].astype(str)

        out_name = filename.replace(".csv", ".parquet")
        out_path = os.path.join(PROCESSED_DIR, out_name)

        try:
            df.to_parquet(out_path, index=False)
            tqdm.write(f"Saved: {out_path}")
        except Exception as e:
            tqdm.write(f"Failed to save {out_path}: {e}")

        # cleanup to keep memory low
        del df
        gc.collect()

    print("STREAMING PIPELINE COMPLETED.")


def cli():
    parser = argparse.ArgumentParser(prog="ml.pipeline_stream")
    parser.add_argument("--n_files", type=int, default=None, help="Number of CSVs to process (default=all)")
    parser.add_argument("--random", action="store_true", help="Sample random files instead of first N")
    args = parser.parse_args()

    run_streaming_pipeline(n_files=args.n_files, sample_random=args.random)


if __name__ == "__main__":
    cli()
