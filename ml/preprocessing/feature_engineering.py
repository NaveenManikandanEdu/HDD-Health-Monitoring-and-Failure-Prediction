# ml/preprocessing/feature_engineering.py
import os, json
import numpy as np
import pandas as pd

ID_COL = "device_id"
WINDOW = 7

CURATED_SENSOR_COLUMNS = [
    "smart_1_raw","smart_5_raw","smart_187_raw","smart_188_raw",
    "smart_193_raw","smart_197_raw","smart_198_raw","smart_199_raw",
    "smart_240_raw","smart_241_raw","smart_242_raw",
    "temperature","power_on_hours"
]

ARTIFACT_DIR = os.path.join("ml", "artifacts")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.json")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def compute_features_for_file(df: pd.DataFrame):
    feats = {}
    grp = df.groupby(ID_COL, sort=False)

    for col in [c for c in CURATED_SENSOR_COLUMNS if c in df.columns]:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)

        feats[f"{col}_last"] = s
        feats[f"{col}_delta"] = grp[col].diff().fillna(0).astype(np.float32)

        r = grp[col].rolling(WINDOW, min_periods=1)
        feats[f"{col}_mean_7"] = r.mean().reset_index(level=0, drop=True).astype(np.float32)
        feats[f"{col}_std_7"]  = r.std().reset_index(level=0, drop=True).fillna(0).astype(np.float32)
        feats[f"{col}_min_7"]  = r.min().reset_index(level=0, drop=True).astype(np.float32)
        feats[f"{col}_max_7"]  = r.max().reset_index(level=0, drop=True).astype(np.float32)

    feat_df = pd.DataFrame(feats)

    if not os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "w") as f:
            json.dump({"features": list(feat_df.columns)}, f, indent=2)

    return feat_df
