from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any
import pandas as pd
import io
import os
import uuid
import joblib
import json
from datetime import datetime

# --- RELATIVE IMPORTS ---
from .evaluate_csv import evaluate_dataframe
from .generate_pdf_report import create_pdf_report
from .db import get_db, init_db
from .auth import Hash, create_access_token, get_current_user
from .state import ML_MODELS, WAREHOUSE_STATE
from controller.core.loader import ModelLoader

# --- DYNAMIC CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models", "risk_7d")

def load_dual_models():
    """Loads the model and returns it for the registry."""
    if not os.path.exists(MODEL_DIR):
        return None, "No Model Dir"
    
    # Load the latest version
    models = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("model_v")],
                    key=lambda x: int(x.split("_v")[1].split(".")[0]) if "_v" in x else 0)
    
    if not models: 
        return None, "No Model File"
        
    model_name = models[-1]
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"📦 Loading Model: {model_name}...")
    
    model = joblib.load(model_path)
    if hasattr(model, "params"): 
        model.params["device"] = "cpu"
    return model, model_name

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Find the REAL active model promoted by the Controller
    model = ModelLoader.load("risk_7d")
    name = ModelLoader.get_active_version("risk_7d")
    
    ML_MODELS["risk_model"] = model
    ML_MODELS["model_name"] = name or "Bootstrap_Model"
    ML_MODELS["health"] = model 
    
    print(f"✅ System Ready: Active Model ({name})")
    yield
    ML_MODELS.clear()

# --- THE GLOBAL APP OBJECT ---
app = FastAPI(title="HDD Risk API", lifespan=lifespan)

# Add CORS so Streamlit can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class WarehousePayload(BaseModel):
    warehouse_id: str
    vault_id: str
    timestamp: str
    records: List[Dict[str, Any]]

# --- AUTH ROUTES ---
@app.post("/auth/register")
def register(user: UserRegister):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE email = ?", (user.email,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = Hash.bcrypt(user.password)
    cur.execute("INSERT INTO users (email, password_hash, is_verified) VALUES (?, ?, 1)", 
                (user.email, hashed_pw))
    conn.commit()
    conn.close()
    return {"message": "Account created successfully."}

@app.post("/auth/login")
def login(user: UserLogin):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE email = ?", (user.email,))
    row = cur.fetchone()
    conn.close()
    
    if not row or not Hash.verify(user.password, row[0]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"access_token": create_access_token({"sub": user.email}), "token_type": "bearer"}

# --- LIVE SIMULATOR LOGIC ---
def save_warnings_background(critical_df, payload):
    """Background task to save alerts without slowing down dashboard."""
    conn = get_db()
    for _, row in critical_df.iterrows():
        risk = row.get('risk_score', 0)
        health = row.get('health_score', 100)
        severity = "CRITICAL" if (risk >= 75 or health <= 50) else "WARNING"

        conn.execute("""
            INSERT INTO warnings (timestamp, warehouse, vault, device_id, risk_score, health_score, mechanism, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.timestamp, 
            payload.warehouse_id, 
            payload.vault_id, 
            str(row.get('device_id', 'Unknown')), 
            float(risk), 
            float(health), 
            str(row.get('dominant_mechanism', 'Unknown')), 
            severity
        ))
    conn.commit()
    conn.close()

@app.get("/health")
def health_check():
    """Handshake endpoint for Tester and Controller."""
    return {
        "status": "ok",
        "models_loaded": {
            "health": ML_MODELS.get("model_name", "None"),
            "risk_7d": ML_MODELS.get("model_name", "None")
        }
    }

# --- REPLACE THIS FUNCTION in web_app/backend/main.py ---

@app.post("/live/update")
def receive_telemetry(payload: WarehousePayload, background_tasks: BackgroundTasks):
    model = ML_MODELS.get("risk_model")
    if not model: 
        return {"error": "ML Model not loaded on server."}

    try:
        # 1. Load Raw Data
        df = pd.DataFrame(payload.records)
        
        # 2. Run Evaluation (The Model adds 'risk_score' and 'health_score' here)
        summary, agg_df = evaluate_dataframe(df, model) 

        # 3. ROBUST COUNTING LOGIC (The Fix)
        # We must use 'agg_df' (processed), NOT 'df' (raw).
        # We also check for 'risk_score' which is the standard name produced by evaluate_dataframe.
        
        if 'risk_score' in agg_df.columns:
            # Count rows where Risk > 50 OR Health < 50
            critical_mask = (agg_df['risk_score'] > 50) | (agg_df['health_score'] < 50)
            actual_critical_count = int(critical_mask.sum())
        else:
            # Fallback if model fails (prevents crash)
            print("⚠️ Warning: Model did not return risk_score. Defaulting to 0.")
            actual_critical_count = 0

        # 4. Update In-Memory Dashboard State
        warehouse = WAREHOUSE_STATE.setdefault(payload.warehouse_id, {})
        
        warehouse[payload.vault_id] = {
            "last_updated": datetime.now().strftime("%H:%M:%S"),
            "total_drives": len(df),
            "critical_count": actual_critical_count, 
            "buckets": summary["buckets"],
        }

        # 5. Handle Critical Alerts in Background
        if not agg_df.empty:
            risky_drives = agg_df[(agg_df["risk_score"] > 50) | (agg_df["health_score"] < 85)]
            if not risky_drives.empty:
                background_tasks.add_task(save_warnings_background, risky_drives, payload)

        # Print confirmation
        print(f"⚡ Update: {payload.vault_id} | Critical: {actual_critical_count}/{len(df)}")
        
        return {"status": "processed", "vault": payload.vault_id}
        
    except Exception as e:
        print(f"❌ Error in telemetry: {e}")
        raise HTTPException(status_code=400, detail=f"Data processing error: {e}")

@app.get("/live/state")
def get_live_state():
    return WAREHOUSE_STATE

@app.get("/live/warnings")
def get_live_warnings():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT timestamp, warehouse, vault, device_id, risk_score, severity 
            FROM warnings ORDER BY id DESC LIMIT 50
        """).fetchall()
        conn.close()
        return [{"time": r[0], "warehouse": r[1], "vault": r[2], "device": r[3], "risk": r[4], "severity": r[5]} for r in rows]
    except:
        return []

# --- BATCH CSV ROUTES ---
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    model = ML_MODELS.get("risk_model")
    if not model: 
        raise HTTPException(status_code=500, detail="ML Engine offline.")

    summary, agg = evaluate_dataframe(df, model)

    buffer = io.BytesIO()
    agg.to_parquet(buffer)
    
    report_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO reports (id, user_email, filename, summary, full_data) VALUES (?, ?, ?, ?, ?)",
        (report_id, current_user, file.filename, json.dumps(summary), buffer.getvalue())
    )
    conn.commit()
    conn.close()

    return {"report_id": report_id, "summary": summary}

@app.get("/reports/history")
def get_history(current_user: str = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("""
        SELECT id, filename, created_at, summary 
        FROM reports WHERE user_email = ? ORDER BY created_at DESC
    """, (current_user,)).fetchall()
    conn.close()
    
    return [{
        "report_id": r[0], 
        "filename": r[1], 
        "date": r[2],
        "total_drives": json.loads(r[3]).get("total_drives", 0), 
        "critical_count": len(json.loads(r[3]).get("top_critical", []))
    } for r in rows]

@app.get("/generate-pdf/{report_id}")
def generate_pdf(report_id: str):
    conn = get_db()
    row = conn.execute("SELECT filename, created_at, full_data FROM reports WHERE id = ?", (report_id,)).fetchone()
    conn.close()
    
    if not row: 
        raise HTTPException(status_code=404, detail="Report not found")
        
    agg_df = pd.read_parquet(io.BytesIO(row[2]))
    pdf_bytes = create_pdf_report(agg_df, model_name=ML_MODELS.get("model_name", "Enterprise_V1"))
    
    filename = f"Report_{row[0]}_{row[1][:10]}.pdf"
    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )