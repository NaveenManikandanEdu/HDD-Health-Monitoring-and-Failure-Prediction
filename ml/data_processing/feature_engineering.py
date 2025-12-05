import os
import json
import gc
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS = os.path.join(BASE, "ml", "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

FEATURE_FILE = os.path.join(ARTIFACTS, "feature_columns.json")


def _load_schema():
    if not os.path.exists(FEATURE_FILE):
        return {"raw": [], "features": []}

    try:
        with open(FEATURE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"raw": [], "features": []}


def _save_schema(schema):
    temp = FEATURE_FILE + ".tmp"
    with open(temp, "w") as f:
        json.dump(schema, f, indent=2)
    os.replace(temp, FEATURE_FILE)


def _detect_raw_columns(df: pd.DataFrame):
    """Any SMART raw column ends with '_raw'."""
    return sorted([c for c in df.columns if c.endswith("_raw")])


def engineer_features(df: pd.DataFrame, lazy=True) -> pd.DataFrame:
    """
    FAST VECTORISED FEATURE ENGINEERING:

    - Detects SMART raw columns
    - Merges with schema
    - Computes delta/min/max for ALL columns in one vectorized groupby
    - Avoids column duplication
    - Avoids DataFrame fragmentation
    """

    if df is None or df.empty:
        return df

    # sort + index reset ensures alignment for concat
    df = df.sort_values(["device_id", "date"]).reset_index(drop=True)

    schema = _load_schema()
    saved_raw = schema.get("raw", [])

    # detect new raw SMART columns
    present_raw = _detect_raw_columns(df)

    # union of schema and current file
    merged_raw = sorted(set(saved_raw) | set(present_raw))

    # extract only columns present in df
    raw_cols = [col for col in merged_raw if col in df.columns]
    if not raw_cols:
        _save_schema({"raw": merged_raw, "features": []})
        return df

    # Create clean numeric-only block
    raw_block = df[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    # -------------------------
    # 🚀 VECTORISED OPERATIONS
    # -------------------------

    group = df.groupby("device_id", sort=False)

    # Delta for all columns in one shot
    delta_block = group[raw_cols].diff().fillna(0).astype("float32")
    delta_block.columns = [f"{c}_delta" for c in raw_cols]

    # Min values for all columns in one shot
    min_block = group[raw_cols].transform("min").astype("float32")
    min_block.columns = [f"{c}_min" for c in raw_cols]

    # Max values for all columns in one shot
    max_block = group[raw_cols].transform("max").astype("float32")
    max_block.columns = [f"{c}_max" for c in raw_cols]

    # --------------------------------------
    # Build final engineered block
    # --------------------------------------
    engineered_df = pd.concat(
        [raw_block, delta_block, min_block, max_block],
        axis=1
    )

    # Drop duplicates if they happen to exist
    duplicate_cols = [c for c in engineered_df.columns if c in df.columns]
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)

    # Final merge (single concat → no fragmentation)
    df = pd.concat([df, engineered_df], axis=1)

    # Save schema
    full_feature_list = list(engineered_df.columns)
    _save_schema({"raw": merged_raw, "features": sorted(full_feature_list)})

    gc.collect()
    return df
