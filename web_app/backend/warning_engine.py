# --- RELATIVE IMPORT FIX ---
from .db import get_db

def generate_warnings(warehouse_id, vault_id, summary, timestamp):
    """
    Checks the evaluation summary for critical/warning drives
    and persists them to the SQLite database.
    """
    conn = get_db()
    cur = conn.cursor()

    # 'top_critical' contains the drives with the highest risk scores
    for d in summary.get("top_critical", []):
        risk = d.get("risk_score", 0)
        health = d.get("health_score", 100)

        # Logic to determine Severity based on Enterprise thresholds
        if risk >= 75 or health <= 50:
            severity = "CRITICAL"
        elif risk >= 15 or health <= 85:
            severity = "WARNING"
        else:
            continue # Skip healthy drives

        # Data Persistence for the Frontend Event Log
        cur.execute("""
        INSERT INTO warnings 
        (timestamp, warehouse, vault, device_id, risk_score, health_score, mechanism, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            warehouse_id,
            vault_id,
            d["device_id"],
            risk,
            health,
            d.get("dominant_mechanism", "Unknown"),
            severity
        ))

    conn.commit()
    conn.close()