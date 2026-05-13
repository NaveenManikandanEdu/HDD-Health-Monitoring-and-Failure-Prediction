import os
import json
import gc
import io
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

# --- IMPORTS ---
from ml.preprocessing.clean import basic_clean
from ml.preprocessing.feature_engineering import compute_features_for_file

# --- CONFIG ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(BASE, "ml", "artifacts")
SCHEMA_PATH = os.path.join(BASE, "ml", "artifacts", "feature_columns.json")

if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH) as f:
        FEATURE_COLS = json.load(f)["features"]
else:
    print(f"⚠️ Warning: Schema not found at {SCHEMA_PATH}.")
    FEATURE_COLS = []

def assign_bucket(score):
    """Categorizes drives based on health score (0-100)."""
    if score >= 85: return "Very Good Health"
    if score >= 60: return "Medium Health"
    if score >= 40: return "Lesser Health"
    if score >= 20: return "Bad Health"
    return "Critical Health"

# --- WORKER: Parallel Processing ---
def process_partition(df_chunk):
    try:
        df_chunk = basic_clean(df_chunk)
    except Exception:
        if "device_id" not in df_chunk.columns and "serial_number" in df_chunk.columns:
            df_chunk["device_id"] = df_chunk["serial_number"]
            
    feat_chunk = compute_features_for_file(df_chunk)
    feat_chunk.index = df_chunk.index
    return df_chunk, feat_chunk

# --- RICH CONTENT GENERATOR ---
# --- In evaluate_csv.py ---

def generate_detailed_stats(agg):
    stats = {}
    # Including "Very Good Health" can help verify IDs are working
    buckets_to_process = ["Critical Health", "Bad Health", "Lesser Health", "Medium Health", "Very Good Health"]
    
    for bucket_name in buckets_to_process:
        bucket_data = agg[agg["bucket"] == bucket_name]
        if len(bucket_data) == 0: continue
            
        b_stat = {
            "count": int(len(bucket_data)),
            "avg_score": int(bucket_data["health_score"].mean()),
            "batches": []
        }
        
        mechanisms = bucket_data["dominant_mechanism"].value_counts().head(5) # Show more clusters
        batch_id = 1
        
        for mech_name, count in mechanisms.items():
            sub_group = bucket_data[bucket_data["dominant_mechanism"] == mech_name]
            
            # REMOVE the strict filter to ensure we see IDs
            # if len(sub_group) < 2 and len(bucket_data) > 20: continue 

            sub_avg = sub_group["health_score"].mean()
            pct = round((count / len(bucket_data)) * 100, 2)
            
            # ENSURE we use the correct column for IDs
            sample_ids = sub_group["device_id"].head(15).tolist()

            b_stat["batches"].append({
                "id": f"{bucket_name[0]}{batch_id}",
                "label": f"{mech_name} Cluster",
                "avg_score": int(sub_avg),
                "dominant_pct": pct,
                "mechanism": mech_name,
                "samples": sample_ids  # The IDs for the UI
            })
            batch_id += 1
        
        stats[bucket_name] = b_stat
    return stats

# --- MAIN EVALUATION PIPELINE ---
def evaluate_dataframe(raw_df: pd.DataFrame, model):
    # 1. Scaling Parallelism
    n_cores = multiprocessing.cpu_count()
    if len(raw_df) > 5000 and n_cores > 1:
        id_col = "serial_number" if "serial_number" in raw_df.columns else "device_id"
        if id_col in raw_df.columns:
            unique_ids = raw_df[id_col].unique()
            id_chunks = np.array_split(unique_ids, n_cores)
            df_chunks = [raw_df[raw_df[id_col].isin(ids)] for ids in id_chunks if len(ids) > 0]
        else:
            df_chunks = np.array_split(raw_df, n_cores)

        results = Parallel(n_jobs=n_cores)(delayed(process_partition)(chunk) for chunk in df_chunks)
        cleaned_chunks, feat_chunks = zip(*results)
        raw_df = pd.concat(cleaned_chunks)
        feat_df = pd.concat(feat_chunks)
    else:
        raw_df, feat_df = process_partition(raw_df)
    
    X = feat_df[FEATURE_COLS].fillna(0)

    # 2. Inference
    if hasattr(model, "n_jobs"): 
        model.n_jobs = -1
    
    is_lgbm_booster = "Booster" in str(type(model))
    
    if hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = -1 * model.decision_function(X)
    elif is_lgbm_booster:
        try: 
            raw_scores = model.predict(X, raw_score=True)
        except Exception: 
            raw_scores = model.predict(X)
    else:
        raw_scores = model.predict(X)

    # 3. Z-Score Calibration
    results = pd.DataFrame({"device_id": raw_df["device_id"], "_raw_score": raw_scores})
    idx = results.groupby("device_id")["_raw_score"].idxmax()
    agg = results.loc[idx].copy().reset_index(drop=True)

    raw_agg = agg["_raw_score"]
    z_scores = (raw_agg - raw_agg.mean()) / (raw_agg.std() + 1e-9)
    shifted_z = z_scores - 1.5
    probs = 1 / (1 + np.exp(-shifted_z))
    
    agg["risk_score"] = (probs * 100).round(1)
    agg["health_score"] = (100 - agg["risk_score"]).clip(0, 100).round(1)
    agg["bucket"] = agg["health_score"].apply(assign_bucket)

    # 4. Mechanism Diagnosis
    feature_map_patterns = {
        "smart_5": "Sector Decay", "smart_187": "Read/Write Instability", 
        "smart_197": "Pending Sector Collapse", "smart_198": "Offline Sector Decay",
        "smart_188": "Timeout/Connection Drop", "smart_199": "Interface CRC Error",
        "smart_4": "Mechanical/Spin Struggle", "smart_7": "Seek Error Instability", 
        "smart_9": "Aging/Runtime Fatigue", "smart_194": "Thermal Stress"
    }
    
    col_to_name = {}
    for col in FEATURE_COLS:
        mapped_name = "Operational Instability"
        for pattern, name in feature_map_patterns.items():
            if pattern in col: 
                mapped_name = name
                break
        col_to_name[col] = mapped_name

    X_agg = X.loc[agg.index].copy()
    X_norm = (X_agg - X_agg.mean()) / (X_agg.std() + 1e-9)
    agg["dominant_mechanism"] = X_norm.idxmax(axis=1).map(col_to_name)
    agg.loc[agg["health_score"] >= 85, "dominant_mechanism"] = "Stable"

    # 5. Build Summary Data Packet
    top_50 = agg.sort_values("risk_score", ascending=False).head(50)
    top_5_risky = agg.sort_values("risk_score", ascending=False).head(5)

    summary = {
        "total_drives": int(len(agg)),
        "avg_fleet_health": float(agg["health_score"].mean()) if not agg.empty else 0.0,
        "buckets": agg["bucket"].value_counts().to_dict(),
        "top_critical": top_50.to_dict(orient="records"),
        "short_term_risk": top_5_risky.to_dict(orient="records"),
        "detailed_stats": generate_detailed_stats(agg),
        "config": {
            "method": "Pattern Similarity Analysis (Z-Calibrated)",
            "horizon": "~7 days"
        }
    }

    gc.collect()
    return summary, agg