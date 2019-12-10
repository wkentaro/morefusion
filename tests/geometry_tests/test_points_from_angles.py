import numpy as np

from morefusion.geometry import points_from_angles


def test_points_from_angles():
    distance = [1]
    elevation = [30]
    azimuth = [45]

    point1 = points_from_angles(
        distance=distance,
        elevation=elevation,
        azimuth=azimuth,
        is_degree=True,
    )
    assert point1.shape == (1, 3)
    assert point1.dtype == np.float64

    point2 = points_from_angles(
        distance=distance,
        elevation=np.deg2rad(elevation),
        azimuth=np.deg2rad(azimuth),
        is_degree=False,
    )
    assert point2.shape == (1, 3)
    assert point2.dtype == np.float64

    np.testing.assert_allclose(point1, point2)
