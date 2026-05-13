import glob, os, pandas as pd

META_COLS = {"device_id", "date", "failure", "failure_next_day", "failure_next_7_days"}

def list_parquets(dir_path):
    return sorted(glob.glob(os.path.join(dir_path, "*.parquet")))

def load_parquet(path):
    df = pd.read_parquet(path)
    features = [c for c in df.columns if c not in META_COLS]
    return df, features