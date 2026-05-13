import os
import pandas as pd

ID_COL, DATE_COL, LABEL_COL = "device_id", "date", "failure"

def basic_clean(df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    # Standardize ID [cite: 35]
    if ID_COL not in df.columns and "serial_number" in df.columns:
        df[ID_COL] = df["serial_number"].astype(str)
    
    df[ID_COL] = df[ID_COL].astype(str)
    df[DATE_COL] = pd.to_datetime(df.get(DATE_COL), errors="coerce")
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)
    return df.drop_duplicates()

def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)