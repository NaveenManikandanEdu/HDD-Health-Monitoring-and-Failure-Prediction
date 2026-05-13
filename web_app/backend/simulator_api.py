import pandas as pd
from fastapi import APIRouter
from datetime import datetime

# --- RELATIVE IMPORTS ---
# Using the '.' notation ensures Python looks within the current 'backend' package.
from .evaluate_csv import evaluate_dataframe
from .warning_engine import generate_warnings
from .state import ML_MODELS, WAREHOUSE_STATE

router = APIRouter()

@router.post("/simulate")
def simulate(payload: dict):
    # 1. Convert Payload to DataFrame
    try:
        # Check if "records" exists to avoid KeyErrors
        records = payload.get("records", [])
        if not records:
            return {"error": "No records found in payload"}
        df = pd.DataFrame(records)
    except Exception as e:
        return {"error": f"Invalid data format: {str(e)}"}

    # 2. Get the Loaded ML Model
    model = ML_MODELS.get("risk_model")
    if not model:
        return {"error": "Model not loaded on backend. Please check server logs."}

    # 3. Run ML Evaluation
    # We ignore the second return (full_df) for the live simulator to save memory
    summary, _ = evaluate_dataframe(df, model)

    # 4. Update In-Memory State (For Live Dashboard)
    wid = payload.get("warehouse_id", "Unknown")
    vid = payload.get("vault_id", "Unknown")
    ts = payload.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

    # Update the global thread-safe state
    WAREHOUSE_STATE.setdefault(wid, {})[vid] = {
        "summary": summary,
        "last_updated": ts,
        "total_drives": summary.get("total_drives", 0),
        "critical_count": len(summary.get("top_critical", [])),
        "buckets": summary.get("buckets", {})
    }

    # 5. Generate & Save Warnings to Database
    # This persists critical events (e.g., Sector Decay) to the SQLite DB
    try:
        generate_warnings(wid, vid, summary, ts)
    except Exception as e:
        print(f"⚠️ Warning Engine Error: {e}")

    return {"status": "ok", "processed_drives": summary.get("total_drives")}