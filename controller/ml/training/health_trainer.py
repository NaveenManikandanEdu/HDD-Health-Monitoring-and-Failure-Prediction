from tqdm import tqdm
import lightgbm as lgb
import joblib
import numpy as np
from pathlib import Path
from controller.config import MODELS_DIR
from controller.ml.utils.dataset import list_parquets, load_parquet


def compute_health_target(df):
    cols = [c for c in df.columns if "temp" in c.lower()]
    if not cols:
        return np.zeros(len(df))

    raw = df[cols].abs().mean(axis=1)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)


def _next_version(directory: Path):
    models = sorted(directory.glob("model_v*.pkl"))
    return len(models) + 1


def train_health(processed_dir: str):
    archive_dir = MODELS_DIR / "archive" / "health"
    active_dir = MODELS_DIR / "active" / "health"

    archive_dir.mkdir(parents=True, exist_ok=True)

    active_models = sorted(active_dir.glob("model_v*.pkl"))
    booster = joblib.load(active_models[-1]) if active_models else None

    files = list_parquets(processed_dir)

    for f in tqdm(files, desc="Health Training", unit="file"):
        df, feats = load_parquet(f)
        X = df[feats].fillna(0)
        y = compute_health_target(df)

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "verbosity": -1
        }

        booster = lgb.train(
            params,
            lgb.Dataset(X, y),
            num_boost_round=300,
            init_model=booster
        )

    version = _next_version(archive_dir)
    model_path = archive_dir / f"model_v{version}.pkl"
    joblib.dump(booster, model_path)

    return model_path