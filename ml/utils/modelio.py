import os, joblib, json
from datetime import datetime

def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load(path):
    return joblib.load(path)

def latest(dir):
    models = sorted(
        [f for f in os.listdir(dir) if f.endswith(".pkl")],
        key=lambda x: int(x.split("_v")[1].split(".")[0])
    )
    return load(os.path.join(dir, models[-1]))

def metadata(path, info):
    with open(path, "w") as f:
        json.dump(info | {"time": datetime.utcnow().isoformat()}, f, indent=2)
