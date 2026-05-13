from pathlib import Path

# BASE_DIR is C:\hdd2\controller
BASE_DIR = Path(__file__).resolve().parent

# Storage Structure
STORAGE_DIR = BASE_DIR / "storage"
ARCHIVE_DIR = STORAGE_DIR / "archive"
CSV_DIR = STORAGE_DIR / "csv"
PROCESSED_CSV_DIR = STORAGE_DIR / "processed_csv"
INVALID_DIR = STORAGE_DIR / "invalid"
MODELS_DIR = STORAGE_DIR / "models"
CHECKPOINT_DIR = STORAGE_DIR / "checkpoints"

# ML Artifacts
ARTIFACTS_DIR = STORAGE_DIR / "artifacts"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Create all folders to prevent "File Not Found" errors
for path in [
    ARCHIVE_DIR, CSV_DIR, PROCESSED_CSV_DIR, INVALID_DIR, 
    MODELS_DIR, CHECKPOINT_DIR, ARTIFACTS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)