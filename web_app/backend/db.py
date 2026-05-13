import sqlite3
import os

# Store DB in the web_app/backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")

def get_db():
    """Establishes a connection to the SQLite database."""
    # check_same_thread=False is needed for FastAPI multi-threading
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """Creates the tables if they don't exist. Starts EMPTY."""
    conn = get_db()
    cur = conn.cursor()

    # 1. Users Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        is_verified INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. OTP Verification Table (Optional, for future use)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS otp_verification (
        email TEXT PRIMARY KEY,
        otp TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL
    )
    """)

    # 3. Reports Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        user_email TEXT,
        filename TEXT,
        summary TEXT NOT NULL,
        full_data BLOB, 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 4. Warnings Table (NEW: For Live Simulator Alerts)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS warnings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        warehouse TEXT,
        vault TEXT,
        device_id TEXT,
        risk_score REAL,
        health_score REAL,
        mechanism TEXT,
        severity TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")

if __name__ == "__main__":
    init_db()
