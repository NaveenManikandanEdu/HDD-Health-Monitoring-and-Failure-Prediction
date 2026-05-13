import os, joblib, json
from datetime import datetime

def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def metadata(path, info):
    with open(path, "w") as f:
        json.dump(info | {"time": datetime.utcnow().isoformat()}, f, indent=2)