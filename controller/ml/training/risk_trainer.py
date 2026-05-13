import lightgbm as lgb
import joblib
from controller.config import MODELS_DIR
from controller.ml.utils.dataset import list_parquets, load_parquet

def train_risk(processed_dir: str):
    archive_dir = MODELS_DIR / "archive" / "risk_7d"
    active_dir = MODELS_DIR / "active" / "risk_7d"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Load existing active model as base [cite: 336]
    active_models = sorted(active_dir.glob("model_v*.pkl"))
    booster = joblib.load(active_models[-1]) if active_models else None

    files = list_parquets(processed_dir)
    for f in files:
        df, feats = load_parquet(f)
        X, y = df[feats].fillna(0), df["failure_next_7_days"].astype(int)
        
        if y.sum() < 5: continue
        
        # Incremental training [cite: 338]
        booster = lgb.train(
            {"objective": "binary", "metric": "auc", "verbosity": -1},
            lgb.Dataset(X, y),
            num_boost_round=300,
            init_model=booster
        )

    version = len(list(archive_dir.glob("model_v*.pkl"))) + 1
    model_path = archive_dir / f"model_v{version}.pkl"
    joblib.dump(booster, model_path)
    return model_path