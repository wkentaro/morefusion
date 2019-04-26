import numpy as np


def backproject_pixel(
    u,
    v,
    z,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    return np.dstack((x, y, z))
