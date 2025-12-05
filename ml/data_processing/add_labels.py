"""
ml/data_processing/add_labels.py

Add prediction labels:
 - failure_next_day : whether the same device fails on the next observation
 - failure_next_7_days : whether the same device has >=1 failure in the next 7 observations

Notes:
 - Assumes rows are sorted by device_id and date prior to calling.
 - Uses FixedForwardWindowIndexer to compute forward-looking 7-row window.
"""

from typing import Optional
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    # make sure sorted
    df = df.sort_values(["device_id", "date"]).reset_index(drop=True)

    # next day (shift -1 within device group)
    df["failure_next_day"] = (
        df.groupby("device_id")["failure"].shift(-1).fillna(0).astype(int)
    )

    # next 7 days: forward window of size 7, compute max, then shift -1 to exclude current row (we want future)
    window = FixedForwardWindowIndexer(window_size=7)

    # groupby + rolling with forward window; this produces an index-aligned Series per group
    # Note: rolling(...).max() will return a Series indexed same as original; we then shift(-1) to look ahead
    grouped = df.groupby("device_id")["failure"]
    try:
        rolled = grouped.rolling(window=window, min_periods=1).max()
        # rolled is a Series with a MultiIndex (device_id, original_index). collapse it.
        rolled = rolled.reset_index(level=0, drop=True)
    except Exception:
        # fallback: if rolling with forward indexer not available (older pandas), compute with a manual loop
        def _manual_7d(g):
            arr = g.values
            out = []
            n = len(arr)
            for i in range(n):
                # look ahead up to next 7 rows (i+1 .. i+7)
                end = min(n, i + 1 + 7)
                out.append(int(arr[i+1:end].max() if (end - (i+1)) > 0 else 0))
            return pd.Series(out, index=g.index)

        rolled = grouped.apply(lambda g: g)  # placeholder - we'll override below
        rolled = grouped.apply(lambda g: _manual_7d(g))
        rolled = rolled.reset_index(level=0, drop=True)

    # shift(-1) considered earlier; rolled already looked at windows including current; we need future-only
    # So shift the rolled series by -1 to align so current row sees the next-window result
    df["failure_next_7_days"] = rolled.shift(-1).fillna(0).astype(int)

    return df
