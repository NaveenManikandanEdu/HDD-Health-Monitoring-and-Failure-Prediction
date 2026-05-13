import os, shutil
from controller.config import ARCHIVE_DIR, CSV_DIR, PROCESSED_CSV_DIR, INVALID_DIR
from controller.core.logger import log
from controller.core.validator import validate_csv
from controller.core.evaluation_pipeline import evaluate_csv
from controller.core.trainer import run_training_cycle

# Absolute Imports to prevent "ghost" ml folders
from controller.ml.preprocessing.preprocess import (
    build_timelines_with_checkpoint, 
    process_raws_with_checkpoint, 
    finalize_labels
)
from controller.ml.preprocessing.add_labels import build_label_map_from_timelines

def process_archive_file(csv_path):
    """Trigger 1: Evaluation (1 file)."""
    if not csv_path.exists(): return
    log(f"Processing archive file: {csv_path.name}")
    valid, msg = validate_csv(csv_path)
    if not valid:
        shutil.move(str(csv_path), str(INVALID_DIR / csv_path.name))
        return
    try:
        evaluate_csv(csv_path)
        if csv_path.exists():
            staged_path = CSV_DIR / csv_path.name
            shutil.move(str(csv_path), str(staged_path))
            log(f"Staged {csv_path.name} for 7-day batch.")
            check_and_trigger_batch()
    except Exception as e:
        log(f"Archive processing failed: {e}", "ERROR")

def check_and_trigger_batch():
    """Trigger 2: 7 CSVs -> Preprocessing -> Trigger Training [cite: 18-22]."""
    csv_files = [str(CSV_DIR / f) for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    
    if len(csv_files) >= 7:
        log(f"Batch threshold met ({len(csv_files)}/7). Starting Preprocessing...")
        try:
            # Phase 1, 2, and 3 [cite: 328-337]
            timelines = build_timelines_with_checkpoint(csv_files)
            date_label_map = build_label_map_from_timelines(timelines)
            process_raws_with_checkpoint(csv_files)
            finalize_labels(date_label_map)

            log("✅ Preprocessing complete. Clearing CSV staging buffer.")
            for f in csv_files: os.remove(f)
            
            # TRIGGER 3: Don't wait! Check for training now.
            check_training_trigger()
        except Exception as e:
            log(f"Preprocessing FAILED: {e}", "ERROR")

def check_training_trigger():
    """Checks if we have 7 Parquets to start the training cycle."""
    parquets = list(PROCESSED_CSV_DIR.glob("*.parquet"))
    log(f"Training Check: {len(parquets)}/7 Parquets ready.")
    
    if len(parquets) >= 7:
        log("Triggering Incremental Training Cycle...")
        from controller.core.trainer import run_training_cycle
        run_training_cycle()

def check_training_trigger():
    """Trigger 3: Training (7 parquets)."""
    parquets = [f for f in os.listdir(PROCESSED_CSV_DIR) if f.endswith(".parquet")]
    if len(parquets) >= 7:
        log(f"Training threshold met ({len(parquets)}/7). Starting Training...")
        run_training_cycle()