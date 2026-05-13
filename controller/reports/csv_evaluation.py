import os
import json
import gc
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

from controller.config import SCHEMA_PATH
from controller.ml.preprocessing.clean import basic_clean
from controller.ml.preprocessing.feature_engineering import compute_features_for_file

# Load schema using the direct config path
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH) as f:
        FEATURE_COLS = json.load(f)["features"]
else:
    FEATURE_COLS = []


# ---------------------------------------------------------
# BUCKET ASSIGNMENT
# ---------------------------------------------------------
def assign_bucket(score):
    if score >= 85: return "Very Good Health"
    if score >= 60: return "Medium Health"
    if score >= 40: return "Lesser Health"
    if score >= 20: return "Bad Health"
    return "Critical Health"


# ---------------------------------------------------------
# FEATURE ENGINEERING WORKER
# ---------------------------------------------------------
def process_partition(df_chunk):
    try:
        df_chunk = basic_clean(df_chunk)
    except Exception as e:
        # Catching the exception securely so it doesn't fail silently
        if "device_id" not in df_chunk.columns and "serial_number" in df_chunk.columns:
            df_chunk["device_id"] = df_chunk["serial_number"]

    feat_chunk = compute_features_for_file(df_chunk)
    feat_chunk.index = df_chunk.index
    return df_chunk, feat_chunk


# ---------------------------------------------------------
# DETAILED CLUSTER STATS (Feeds the PDF)
# ---------------------------------------------------------
def generate_detailed_stats(agg):
    stats = {}
    buckets = [
        "Critical Health",
        "Bad Health",
        "Lesser Health",
        "Medium Health",
        "Very Good Health"
    ]

    for bucket_name in buckets:
        bucket_data = agg[agg["bucket"] == bucket_name]
        if bucket_data.empty:
            continue

        b_stat = {
            "count": int(len(bucket_data)),
            "avg_score": int(bucket_data["health_score"].mean()),
            "batches": []
        }

        mechanisms = bucket_data["dominant_mechanism"].value_counts().head(5)
        batch_id = 1

        for mech_name, count in mechanisms.items():
            sub_group = bucket_data[
                bucket_data["dominant_mechanism"] == mech_name
            ]

            sub_avg = sub_group["health_score"].mean()
            # FIX 1: Use round instead of int to prevent 0% prevalence
            pct = round((count / len(bucket_data)) * 100, 2)

            sample_ids = sub_group["device_id"].head(15).tolist()

            b_stat["batches"].append({
                "id": f"{bucket_name[0]}{batch_id}",
                "label": f"{mech_name} Cluster",
                "avg_score": int(sub_avg),
                "dominant_pct": pct,
                "mechanism": mech_name,
                "samples": sample_ids
            })

            batch_id += 1

        stats[bucket_name] = b_stat

    return stats


# ---------------------------------------------------------
# MAIN ML EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_csv_dataframe(raw_df: pd.DataFrame, model):

    n_cores = multiprocessing.cpu_count()

    # -----------------------------
    # Parallel Feature Engineering
    # -----------------------------
    if len(raw_df) > 5000 and n_cores > 1:
        id_col = "serial_number" if "serial_number" in raw_df.columns else "device_id"

        unique_ids = raw_df[id_col].unique()
        id_chunks = np.array_split(unique_ids, n_cores)

        df_chunks = [
            raw_df[raw_df[id_col].isin(ids)]
            for ids in id_chunks if len(ids) > 0
        ]

        results = Parallel(n_jobs=n_cores)(
            delayed(process_partition)(chunk) for chunk in df_chunks
        )

        cleaned_chunks, feat_chunks = zip(*results)
        raw_df = pd.concat(cleaned_chunks)
        feat_df = pd.concat(feat_chunks)

    else:
        raw_df, feat_df = process_partition(raw_df)

    X = feat_df[FEATURE_COLS].fillna(0)

    # -----------------------------
    # Model Inference
    # -----------------------------
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
        except TypeError: # Specifically catch LightGBM signature errors
            raw_scores = model.predict(X)
    else:
        raw_scores = model.predict(X)

    # -----------------------------
    # Z-Score Calibration
    # -----------------------------
    results = pd.DataFrame({
        "device_id": raw_df["device_id"],
        "_raw_score": raw_scores
    })

    idx = results.groupby("device_id")["_raw_score"].idxmax()
    agg = results.loc[idx].copy().reset_index(drop=True)

    raw_agg = agg["_raw_score"]
    z_scores = (raw_agg - raw_agg.mean()) / (raw_agg.std() + 1e-9)
    shifted_z = z_scores - 1.5
    probs = 1 / (1 + np.exp(-shifted_z))

    agg["risk_score"] = (probs * 100).round(1)
    agg["health_score"] = (100 - agg["risk_score"]).clip(0, 100).round(1)
    agg["bucket"] = agg["health_score"].apply(assign_bucket)

    # -----------------------------
    # Dominant Mechanism
    # -----------------------------
    feature_map_patterns = {
        "smart_5": "Sector Decay",
        "smart_187": "Read/Write Instability",
        "smart_197": "Pending Sector Collapse",
        "smart_198": "Offline Sector Decay",
        "smart_188": "Timeout/Connection Drop",
        "smart_199": "Interface CRC Error",
        "smart_4": "Mechanical/Spin Struggle",
        "smart_7": "Seek Error Instability",
        "smart_9": "Aging/Runtime Fatigue",
        "smart_194": "Thermal Stress"
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

    # -----------------------------
    # Summary Packet
    # -----------------------------
    top_50 = agg.sort_values("risk_score", ascending=False).head(50)
    top_5 = agg.sort_values("risk_score", ascending=False).head(5)

    summary = {
        "total_drives": int(len(agg)),
        "avg_fleet_health": float(agg["health_score"].mean()) if not agg.empty else 0.0,
        "buckets": agg["bucket"].value_counts().to_dict(),
        "top_critical": top_50.to_dict(orient="records"),
        "short_term_risk": top_5.to_dict(orient="records"),
        "detailed_stats": generate_detailed_stats(agg),
        "config": {
            "method": "Pattern Similarity Analysis (Z-Calibrated)",
            "horizon": "~7 days"
        }
    }

    gc.collect()
    return summary, agg

# ---------------------------------------------------------
# EMAIL BODY GENERATOR
# ---------------------------------------------------------
def generate_email_text_report(agg):
    lines = []
    lines.append("=" * 80)
    lines.append(" HDD HEALTH ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("ANALYSIS CONFIGURATION")
    lines.append("---------------------")
    lines.append("Evaluation Method        : Pattern Similarity Analysis (Z-Calibrated)")
    lines.append("Processing Mode          : Parallel/Memory Optimized")
    lines.append("Score Range              : 0 - 100")
    lines.append("Short-Term Risk Horizon  : ~7 days")
    lines.append("")
    lines.append("=" * 80)
    lines.append(f" Total HDDs Identified and Analyzed : {len(agg):,}")
    lines.append("=" * 80)
    lines.append("")

    buckets_to_show = ["Medium Health", "Lesser Health", "Bad Health", "Critical Health"]
    for bucket_name in buckets_to_show:
        bucket_data = agg[agg["bucket"] == bucket_name]
        if len(bucket_data) == 0: continue
            
        avg_score = bucket_data["health_score"].mean()
        lines.append("=" * 80)
        lines.append(f" BUCKET: {bucket_name.upper()} (Count: {len(bucket_data):,})")
        lines.append("=" * 80)
        lines.append(f" Average Score : ~{int(avg_score)}\n")
        
        mechanisms = bucket_data["dominant_mechanism"].value_counts().head(3)
        batch_id = 1
        for mech_name, count in mechanisms.items():
            sub_group = bucket_data[bucket_data["dominant_mechanism"] == mech_name]
            if len(sub_group) < 3: continue

            sub_avg = sub_group["health_score"].mean()
            # FIX 2: Use round instead of int to prevent 0% prevalence
            pct = round((count / len(bucket_data)) * 100, 2)
            
            lines.append("-" * 80)
            lines.append(f"Batch Label      : {mech_name} Cluster")
            lines.append(f"Average Score   : {int(sub_avg)}")
            lines.append(f"Prevalence      : {pct}%")
            lines.append("Affected Drive Group (Top 5 IDs):")
            for _, r in sub_group.head(5).iterrows():
                lines.append(f"  {r.device_id}")
            lines.append("")
            batch_id += 1

    lines.append("=" * 80)
    lines.append(" TOP 5 MOST CRITICAL DRIVES (IMMEDIATE ACTION)")
    lines.append("=" * 80)
    lines.append(f"{'Device ID':<15} | {'Health':<7} | {'Risk':<7} | {'Diagnosis'}")
    lines.append("-" * 75)
    
    top_5 = agg.sort_values("risk_score", ascending=False).head(5)
    for _, row in top_5.iterrows():
        lines.append(f"{row.device_id:<15} | {row.health_score:<7.1f} | {row.risk_score:<7.1f} | {row.dominant_mechanism}")

    lines.append("\n" + "=" * 80)
    lines.append(" Please see the attached PDF for full visualizations and top 50 inventory.")
    lines.append("=" * 80)
    
    return "\n".join(lines)