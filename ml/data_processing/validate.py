"""
ml/data_processing/validate.py

Simple validation helper for processed DataFrames.
"""

from typing import Tuple
import pandas as pd
import numpy as np

REQUIRED = ["device_id", "date", "failure"]


def validate_dataframe(df: pd.DataFrame, require_labels: bool = True) -> Tuple[bool, str]:
    if df is None:
        return False, "DataFrame is None"

    for col in REQUIRED:
        if col not in df.columns:
            return False, f"Missing required column: {col}"

    if df[["device_id", "date"]].isna().any().any():
        return False, "device_id/date contains NA values"

    if require_labels:
        labels = ["failure_next_day", "failure_next_7_days"]
        for col in labels:
            if col not in df.columns:
                return False, f"Missing label column: {col}"
            # check not all NaN
            if df[col].isna().all():
                return False, f"Label column {col} is all NaN"

    return True, "Validation passed"
