"""Colour-scale helpers shared by the app and its tests."""
import numpy as np


def score_limits(scores):
    values = scores[np.isfinite(scores)]
    if not len(values):
        return None
    return float(values.min()), float(values.max())
