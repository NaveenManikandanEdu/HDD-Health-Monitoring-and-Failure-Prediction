
# Global Variables to be shared across modules

# Loaded ML models (e.g., {"risk_model": model_obj})
ML_MODELS = {}

# Latest warehouse snapshot (In-Memory Dashboard State)
WAREHOUSE_STATE = {}

# Warning history (append-only list, though we mostly use DB now)
WARNING_HISTORY = []
