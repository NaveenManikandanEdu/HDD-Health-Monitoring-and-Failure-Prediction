"""
ml/data_processing/clean.py

Basic cleaning for raw CSV rows.
- Ensure device_id (fallback to serial_number)
- Ensure date (parsed)
- Ensure failure exists and is integer
- Do NOT drop columns (to avoid schema drift)
- Sort by device_id/date and fill remaining NAs with 0
"""

import os
import pandas as pd

REQUIRED_ID = "device_id"
ALT_ID = "serial_number"
REQUIRED_DATE = "date"
REQUIRED_FAILURE = "failure"


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure device_id (allow serial_number fallback)
    if REQUIRED_ID not in df.columns:
        if ALT_ID in df.columns:
            df = df.rename(columns={ALT_ID: REQUIRED_ID})
        else:
            raise ValueError("Missing required column: device_id or serial_number")

    # Ensure date exists and convert to datetime
    if REQUIRED_DATE not in df.columns:
        raise ValueError("Missing required column: date")
    df[REQUIRED_DATE] = pd.to_datetime(df[REQUIRED_DATE], errors="coerce")

    # Ensure failure column exists
    if REQUIRED_FAILURE not in df.columns:
        df[REQUIRED_FAILURE] = 0
    # Coerce to numeric then to int (fillna with 0)
    df[REQUIRED_FAILURE] = pd.to_numeric(df[REQUIRED_FAILURE], errors="coerce").fillna(0).astype(int)

    # Do not drop columns — keep full schema
    # Sort by device_id and date for deterministic downstream operations
    df = df.sort_values([REQUIRED_ID, REQUIRED_DATE]).reset_index(drop=True)

    # Fill remaining NA values with 0 (safe default)
    df = df.fillna(0)

    return df
