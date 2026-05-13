# controller/core/model_evaluator.py

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from controller.config import MODELS_DIR, PROCESSED_CSV_DIR
from controller.ml.utils.dataset import list_parquets, load_parquet


def evaluate_risk(model_path):
    model = joblib.load(model_path)
    scores = []

    for f in list_parquets(str(PROCESSED_CSV_DIR)):
        df, feats = load_parquet(f)
        X = df[feats].fillna(0)
        y = df["failure_next_7_days"]

        if len(set(y)) < 2:
            continue

        preds = model.predict(X)
        scores.append(roc_auc_score(y, preds))

    return float(np.mean(scores)) if scores else 0.0


def evaluate_health(model_path):
    import joblib
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from controller.config import PROCESSED_CSV_DIR
    from controller.ml.utils.dataset import list_parquets, load_parquet

    model = joblib.load(model_path)
    scores = []

    for f in list_parquets(str(PROCESSED_CSV_DIR)):
        df, feats = load_parquet(f)
        X = df[feats].fillna(0)
        y = df["failure_next_day"]

        preds = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        scores.append(rmse)

    return float(np.mean(scores)) if scores else float("inf")