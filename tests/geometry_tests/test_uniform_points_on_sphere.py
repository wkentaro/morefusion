from morefusion.geometry import uniform_points_on_sphere

import numpy as np


def test_uniform_points_on_sphere():
    points = uniform_points_on_sphere(angle_sampling=10, radius=1)
    assert points.shape == (82, 3)
    assert points.dtype == np.float64
    assert ((-1 <= points) & (points <= 1)).all()
