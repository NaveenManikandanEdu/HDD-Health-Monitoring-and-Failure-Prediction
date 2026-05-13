import os
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ml.utils.dataset import list_parquets, load_parquet
from ml.utils.imbalance import scale_pos_weight
from ml.utils.modelio import save, metadata
from ml.utils.gcutils import clean


def _next_version(out_dir):
    vs = []
    for f in os.listdir(out_dir):
        if f.startswith("model_v"):
            try:
                vs.append(int(f.split("_v")[1].split(".")[0]))
            except Exception:
                pass
    return max(vs) + 1 if vs else 1


def train_risk(processed_dir: str, n_files: int | None = None):
    OUT = "ml/models/risk_7d"
    CKPT = "ml/checkpoints/risk_7d/ckpt.pkl"

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    files = list_parquets(processed_dir)
    if n_files:
        files = files[-n_files:]

    booster = lgb.Booster(model_file=CKPT) if os.path.exists(CKPT) else None
    features = None

    for f in tqdm(files, desc="Risk training", unit="file"):
        df, feats = load_parquet(f)

        if "failure_next_7_days" not in df.columns:
            raise RuntimeError("failure_next_7_days label missing")

        features = features or feats
        X = df[features].fillna(0)
        y = df["failure_next_7_days"].astype(int)

        # 🔧 SAFETY GUARDS
        if y.sum() < 5:
            continue

        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.01,
            "num_leaves": 64,
            "max_depth": 8,
            "min_data_in_leaf": 200,
            "scale_pos_weight": min(scale_pos_weight(ytr), 1000),
            "verbosity": -1,
        }

        booster = lgb.train(
            params,
            lgb.Dataset(Xtr, ytr),
            valid_sets=[lgb.Dataset(Xv, yv)],
            num_boost_round=2000,
            init_model=booster,
            callbacks=[
                lgb.early_stopping(150),
                lgb.log_evaluation(100),
            ],
        )

        save(booster, CKPT)
        clean()

    version = _next_version(OUT)
    model_path = f"{OUT}/model_v{version}.pkl"
    save(booster, model_path)
    metadata(f"{OUT}/meta_v{version}.json", {"model": "risk_7d"})

    return model_path
