import pandas as pd

def to_percentile(scores):
    return pd.Series(scores).rank(pct=True).values * 100
