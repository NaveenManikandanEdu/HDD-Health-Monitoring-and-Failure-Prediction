# ml/preprocessing/clean.py
import os
import pandas as pd

ID_COL = "device_id"
ALT_ID_COL = "serial_number"
DATE_COL = "date"
LABEL_COL = "failure"


def basic_clean(df: pd.DataFrame):
    df.columns = df.columns.str.strip()

    if ID_COL not in df.columns:
        if ALT_ID_COL in df.columns:
            df[ID_COL] = df[ALT_ID_COL].astype(str)
        else:
            raise ValueError("Missing device_id / serial_number")

    df[ID_COL] = df[ID_COL].astype(str)
    df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")

    df[LABEL_COL] = (
        pd.to_numeric(df.get(LABEL_COL, 0), errors="coerce")
        .fillna(0).astype(int).clip(0, 1)
    )

    return df.drop_duplicates()


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
