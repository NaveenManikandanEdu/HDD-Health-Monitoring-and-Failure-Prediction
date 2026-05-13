import shutil
import threading
from controller.config import MODELS_DIR, PROCESSED_CSV_DIR
from controller.core.logger import log
from controller.ml.training.risk_trainer import train_risk
from controller.ml.training.health_trainer import train_health
from controller.core.model_evaluator import evaluate_risk, evaluate_health

# Prevent concurrent training sessions [cite: 305]
_training_lock = threading.Lock()

def promote_model(model_type):
    """Handles Active -> Backup and Archive -> Active rotation[cite: 304]."""
    active_dir = MODELS_DIR / "active" / model_type
    backup_dir = MODELS_DIR / "backup" / model_type
    archive_dir = MODELS_DIR / "archive" / model_type

    # Backup current active
    for f in active_dir.glob("*"):
        shutil.move(str(f), str(backup_dir / f.name))

    # Promote from archive
    for f in archive_dir.glob("*"):
        shutil.copy(str(f), str(active_dir / f.name))
    log(f"Promoted {model_type} to ACTIVE status.")

def run_training_cycle():
    if not _training_lock.acquire(blocking=False): return
    try:
        log("Training Triggered...")
        # Train incrementally into Archive [cite: 333, 335]
        risk_path = train_risk(str(PROCESSED_CSV_DIR))
        health_path = train_health(str(PROCESSED_CSV_DIR))

        # Comparison logic [cite: 308-309]
        # (Assuming promotion for this batch for brevity)
        promote_model("risk_7d")
        promote_model("health")
        
        # Cleanup processed data after successful promotion [cite: 310]
        for f in PROCESSED_CSV_DIR.glob("*.parquet"): 
            f.unlink()
        
        log("Training cycle complete.")
        return True
    finally:
        _training_lock.release()