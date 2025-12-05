# scripts/check_features.py
import os, json, sys
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS = os.path.join(REPO_ROOT, "ml", "artifacts")
FEATURE_FILE = os.path.join(ARTIFACTS, "feature_columns.json")
PROCESSED = os.path.join(REPO_ROOT, "data", "processed")

def load_schema(path):
    if not os.path.exists(path):
        print(f"❌ feature schema not found: {path}")
        return None
    with open(path, "r") as f:
        j = json.load(f)
    # support multiple formats
    if isinstance(j, dict):
        if "feature_columns" in j:
            return list(j["feature_columns"])
        if "features" in j:
            return list(j["features"])
    if isinstance(j, list):
        return j
    print("❌ Unrecognized feature_columns.json format")
    return None

def main():
    feats = load_schema(FEATURE_FILE)
    if not feats:
        print("→ Fix: run your feature_engineering to create ml/artifacts/feature_columns.json")
        sys.exit(2)
    print(f"✔ feature_columns.json loaded — {len(feats)} features (showing first 20):")
    print(feats[:20])

    if not os.path.exists(PROCESSED):
        print(f"❌ processed dir not found: {PROCESSED}")
        sys.exit(2)
    pfiles = sorted([f for f in os.listdir(PROCESSED) if f.endswith(".parquet")])
    if not pfiles:
        print(f"❌ no parquet files in {PROCESSED}")
        sys.exit(2)
    sample = os.path.join(PROCESSED, pfiles[0])
    print(f"✔ found {len(pfiles)} parquet files — sampling: {pfiles[0]}")
    try:
        df = pd.read_parquet(sample)
    except Exception as e:
        print(f"❌ could not read sample parquet {sample}: {e}")
        sys.exit(2)

    missing = [c for c in feats if c not in df.columns]
    extra = [c for c in df.columns if c not in feats and c not in ("device_id","date","failure","failure_next_day","failure_next_7_days")]
    print(f"Sample rows: {len(df)} ; columns in sample: {len(df.columns)}")
    if missing:
        print(f"⚠ Missing {len(missing)} master feature columns in sample (showing up to 20): {missing[:20]}")
    else:
        print("✔ All master features present in sample (or will be filled during alignment).")
    if extra:
        print(f"ℹ Sample has {len(extra)} extra columns (non-master): {extra[:20]}")
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
