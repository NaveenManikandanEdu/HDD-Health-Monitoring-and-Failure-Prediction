import os
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from ml.utils.dataset import list_parquets, load_parquet
from ml.utils.modelio import save, metadata
from ml.utils.gcutils import clean


def compute_health_target(df):
    cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["temp", "read", "write", "lat"])
    ]
    if not cols:
        return np.zeros(len(df))

    raw = df[cols].abs().mean(axis=1)
    # 🔧 NORMALIZATION (CRITICAL)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)


def _next_version(out_dir):
    vs = []
    for f in os.listdir(out_dir):
        if f.startswith("model_v"):
            try:
                vs.append(int(f.split("_v")[1].split(".")[0]))
            except Exception:
                pass
    return max(vs) + 1 if vs else 1


def train_health(processed_dir: str, n_files: int | None = None):
    OUT = "ml/models/health"
    CKPT = "ml/checkpoints/health/ckpt.pkl"

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    files = list_parquets(processed_dir)
    if n_files:
        files = files[-n_files:]

    booster = lgb.Booster(model_file=CKPT) if os.path.exists(CKPT) else None
    features = None

    for f in tqdm(files, desc="Health training", unit="file"):
        df, feats = load_parquet(f)
        features = features or feats

        X = df[features].fillna(0)
        y = compute_health_target(df)

        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 64,
            "max_depth": 8,
            "verbosity": -1,
        }

        booster = lgb.train(
            params,
            lgb.Dataset(X, y),
            num_boost_round=1500,
            init_model=booster,
        )

        save(booster, CKPT)
        clean()

    version = _next_version(OUT)
    model_path = f"{OUT}/model_v{version}.pkl"
    save(booster, model_path)
    metadata(f"{OUT}/meta_v{version}.json", {"model": "health"})

    return model_path
