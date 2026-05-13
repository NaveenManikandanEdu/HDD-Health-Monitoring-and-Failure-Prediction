import numpy as np

def scale_pos_weight(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))
