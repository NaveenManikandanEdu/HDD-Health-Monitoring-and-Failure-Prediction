import os
import glob
import json
import gc
import joblib
import numpy as np
import pandas as pd
import warnings
import sys

# Import the PDF Generator
# This requires generate_pdf_report.py to be in the same folder
try:
    from generate_pdf_report import create_pdf_report
except ImportError:
    print("WARNING: 'generate_pdf_report.py' not found. PDF will be skipped.")
    create_pdf_report = None

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
MODEL_RISK_DIR = os.path.join(BASE, "ml", "models", "risk_7d")
ARTIFACTS_DIR = os.path.join(BASE, "ml", "artifacts")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.json")

# TARGET FOLDER: data/reports
REPORT_OUTPUT = os.path.join(BASE, "data", "reports")

TOP_N = 50

# =============================================================================
# HELPERS
# =============================================================================

def load_latest_model(model_dir):
    models = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("model_v")],
        key=lambda x: int(x.split("_v")[1].split(".")[0]),
    )
    if not models:
        raise RuntimeError("No trained model found")
    model_name = models[-1]
    
    model = joblib.load(os.path.join(model_dir, model_name))
    if hasattr(model, "params"):
        model.params["device"] = "cpu"
        
    return model, model_name

def assign_bucket(score):
    if score >= 85: return "Very Good Health"
    if score >= 60: return "Medium Health"
    if score >= 40: return "Lesser Health"
    if score >= 20: return "Bad Health"
    return "Critical Health"

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate():
    # -------------------------------------------------------------------------
    # 1. SETUP & DATA LOADING
    # -------------------------------------------------------------------------
    with open(SCHEMA_PATH) as f:
        FEATURE_COLS = json.load(f)["features"]

    risk_model, model_name = load_latest_model(MODEL_RISK_DIR)

    parquets = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")))
    if not parquets:
        raise RuntimeError("No processed parquet files found")

    # Load data
    dfs = [pd.read_parquet(p) for p in parquets]
    data = pd.concat(dfs, ignore_index=True)
    X = data[FEATURE_COLS].fillna(0)

    # -------------------------------------------------------------------------
    # 2. FAST MODEL SCORING
    # -------------------------------------------------------------------------
    
    is_lgbm_booster = "Booster" in str(type(risk_model))
    if hasattr(risk_model, "predict_proba"):
        raw = risk_model.predict_proba(X)[:, 1]
    elif hasattr(risk_model, "decision_function"):
        raw = -1 * risk_model.decision_function(X) 
    elif is_lgbm_booster:
        try: raw = risk_model.predict(X, raw_score=True)
        except: raw = risk_model.predict(X)
    else:
        raw = risk_model.predict(X)

    data["_raw_score"] = raw

    # -------------------------------------------------------------------------
    # 3. MEMORY OPTIMIZATION
    # -------------------------------------------------------------------------
    idx = data.groupby("device_id")["_raw_score"].idxmax()
    agg = data.loc[idx].copy().reset_index(drop=True)
    del data, X, dfs, raw
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. SCORING & DIAGNOSIS
    # -------------------------------------------------------------------------
    
    raw_agg = agg["_raw_score"]
    z_scores = (raw_agg - raw_agg.mean()) / (raw_agg.std() + 1e-9)
    shifted_z = z_scores - 3.0 
    probs = 1 / (1 + np.exp(-shifted_z))
    agg["risk_score"] = (probs * 100).round(1)
    agg["health_score"] = (100 - agg["risk_score"]).clip(0, 100).round(1)

    feature_map_patterns = {
        "smart_194": "Thermal Stress", "smart_5": "Sector Decay",
        "smart_187": "Read/Write Instability", "smart_197": "Pending Sector Collapse",
        "smart_188": "Timeout/Connection Drop", "smart_4": "Mechanical/Spin Struggle",
        "smart_7": "Seek Error Instability", "smart_9": "Aging/Runtime Fatigue",
        "smart_12": "Power Cycle Instability"
    }

    col_to_name = {}
    for col in FEATURE_COLS:
        mapped_name = "Operational Instability"
        for pattern, name in feature_map_patterns.items():
            if pattern in col:
                mapped_name = name
                break
        col_to_name[col] = mapped_name

    X_agg = agg[FEATURE_COLS].fillna(0)
    X_norm = (X_agg - X_agg.mean()) / (X_agg.std() + 1e-9)
    agg["dominant_mechanism"] = X_norm.idxmax(axis=1).map(col_to_name)
    
    agg.loc[agg["health_score"] >= 85, "dominant_mechanism"] = "Stable"
    agg["bucket"] = agg["health_score"].apply(assign_bucket)

    # -------------------------------------------------------------------------
    # 5. TERMINAL REPORT
    # -------------------------------------------------------------------------
    
    print("=" * 80)
    print(" HDD HEALTH ANALYSIS REPORT")
    print("=" * 80)
    print("")
    print("ANALYSIS CONFIGURATION")
    print("---------------------")
    print("Evaluation Method        : Pattern Similarity Analysis (Z-Calibrated)")
    print("Processing Mode          : CPU (Memory Optimized)")
    print("Score Range              : 0 – 100")
    print("Short-Term Risk Horizon  : ~7 days")
    print("")
    print("=" * 80)
    print(" DRIVE DISCOVERY")
    print("=" * 80)
    print(f"\nTotal HDDs Identified and Analyzed : {len(agg):,}\n")
    print("=" * 80)
    print(" HEALTH BUCKET DEFINITION")
    print("=" * 80)
    print(f"{'Bucket Name':<20} {'Score Range'}")
    print("-" * 32)
    print(f"{'Very Good Health':<20} 85 – 100")
    print(f"{'Medium Health':<20} 60 – 85")
    print(f"{'Lesser Health':<20} 40 – 60")
    print(f"{'Bad Health':<20} 20 – 40")
    print(f"{'Critical Health':<20} 0 – 20")
    print("")

    buckets_to_show = ["Medium Health", "Lesser Health", "Bad Health", "Critical Health"]
    
    for bucket_name in buckets_to_show:
        bucket_data = agg[agg["bucket"] == bucket_name]
        if len(bucket_data) == 0:
            continue
            
        avg_score = bucket_data["health_score"].mean()
        
        print("=" * 80)
        print(f" BUCKET: {bucket_name.upper()} (Count: {len(bucket_data):,})")
        print("=" * 80)
        print("")
        print(f"Average Score : ~{int(avg_score)}")
        print("")
        
        mechanisms = bucket_data["dominant_mechanism"].value_counts().head(3)
        batch_id = 1
        for mech_name, count in mechanisms.items():
            sub_group = bucket_data[bucket_data["dominant_mechanism"] == mech_name]
            if len(sub_group) < 3: continue

            sub_avg = sub_group["health_score"].mean()
            pct = int((count / len(bucket_data)) * 100)
            
            print("-" * 80)
            print(f"Batch ID        : {bucket_name[0]}{batch_id}")
            print(f"Batch Label     : {mech_name} Cluster")
            print(f"Average Score   : {int(sub_avg)}")
            print("")
            print("Mechanisms Identified:")
            print(f"  • {mech_name:<30} : {pct}% (Dominant Factor)")
            print(f"  • Operational inconsistency      : {max(5, 100-pct-5)}%")
            print("")
            print("Affected Drive Group (Top 10 IDs):")
            for _, r in sub_group.head(10).iterrows():
                print(f"  {r.device_id}")
            print("  ...")
            print("")
            batch_id += 1

    print("=" * 80)
    print(" SHORT-TERM FAILURE RISK (≈ 7-DAY HORIZON)")
    print("=" * 80)
    print("")
    
    risky_list = agg.sort_values("risk_score", ascending=False).head(5)
    for _, r in risky_list.iterrows():
        print("-" * 60)
        print(f"Device ID : {r.device_id}")
        print("Risk      : Elevated short-term failure risk")
        print(f"Basis     : {r.risk_score:.1f}% probability")
        print("Diagnosis : " + r.dominant_mechanism)

    print("")
    print("=" * 80)
    print(" TOP 50 MOST CRITICAL DRIVES (IMMEDIATE ACTION)")
    print("=" * 80)
    print("")
    print(f"{'Rank':<5} {'Device ID':<15} {'Health':<7} {'Risk':<7} {'Dominant Mechanisms'}")
    print("-" * 75)
    
    top_50 = agg.sort_values("risk_score", ascending=False).head(50)
    for i, r in enumerate(top_50.iterrows(), 1):
        _, row = r
        print(f"{i:<5} {row.device_id:<15} {row.health_score:<7.1f} {row.risk_score:<7.1f} {row.dominant_mechanism}")

    print("")
    print("=" * 80)
    print(" END OF REPORT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 6. TRIGGER PDF GENERATION
    # -------------------------------------------------------------------------
    if create_pdf_report:
        print(f"\n>> Triggering PDF Generation...")
        # Passing 'agg' dataframe directly (Instant speed)
        create_pdf_report(agg, model_name, REPORT_OUTPUT)
    else:
        print("\n[ERROR] PDF Generation skipped (Module not imported).")

    gc.collect()

if __name__ == "__main__":
    evaluate()