import os, json
import numpy as np
import pandas as pd
from controller.config import SCHEMA_PATH

ID_COL, WINDOW = "device_id", 7
CURATED_SENSOR_COLUMNS = [
    "smart_1_raw","smart_5_raw","smart_187_raw","smart_188_raw",
    "smart_193_raw","smart_197_raw","smart_198_raw","smart_199_raw",
    "smart_240_raw","smart_241_raw","smart_242_raw","temperature","power_on_hours"
]

def compute_features_for_file(df: pd.DataFrame):
    feats = {}
    grp = df.groupby(ID_COL, sort=False)
    for col in [c for c in CURATED_SENSOR_COLUMNS if c in df.columns]:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32)
        feats[f"{col}_last"] = s
        feats[f"{col}_delta"] = grp[col].diff().fillna(0).astype(np.float32)
        r = grp[col].rolling(WINDOW, min_periods=1)
        # Flatten MultiIndex from rolling ops [cite: 37]
        for stat in ['mean', 'std', 'min', 'max']:
            res = getattr(r, stat)().reset_index(level=0, drop=True)
            feats[f"{col}_{stat}_7"] = res.fillna(0).astype(np.float32)

    feat_df = pd.DataFrame(feats)
    if not os.path.exists(SCHEMA_PATH) and os.path.exists(os.path.dirname(SCHEMA_PATH)):
        with open(SCHEMA_PATH, "w") as f:
            json.dump({"features": list(feat_df.columns)}, f, indent=2)
    return feat_df