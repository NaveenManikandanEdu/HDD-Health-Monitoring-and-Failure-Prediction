# ml/utils/hyperparams.py
"""
LightGBM hyperparameters for streaming incremental training (imbalance-hardened).

Critical: DO NOT use is_unbalance=True in streaming one-file-at-a-time context.
Instead, use a stable GLOBAL_SCALE_POS_WEIGHT to avoid per-file oscillation when files
have zero positives.
"""

LGB_PARAMS = {
    # Core
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",          # main metric to monitor; training logs will show this
    "verbosity": -1,

    # Tree complexity
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "max_depth": -1,

    # Regularization / sampling
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,

    # Learning
    # Use small LR to avoid catastrophic forgetting in streaming updates.
    "learning_rate": 0.01,
    # Do NOT set is_unbalance here. We'll use GLOBAL_SCALE_POS_WEIGHT instead.
    "is_unbalance": False,
    "scale_pos_weight": 1.0,  # will be overridden by training code using GLOBAL_SCALE_POS_WEIGHT
}

# A stable, global scale_pos_weight to apply to every file during streaming training.
# Tweak this as you learn the global imbalance (100.0 is an example; change if needed).
GLOBAL_SCALE_POS_WEIGHT = 100.0

# Default rounds per file (how many boosting rounds to run using each incoming file)
DEFAULT_ROUNDS_PER_FILE = 30

# Learning-rate multiplicative decay applied per file (keeps effective updates small over time).
# e.g., 0.995 -> lr slightly decays each processed file. Set to 1.0 to disable decay.
LR_DECAY_PER_FILE = 0.995
