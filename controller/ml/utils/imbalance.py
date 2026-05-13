def scale_pos_weight(y):
    """Calculates weight for imbalanced LightGBM training."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))