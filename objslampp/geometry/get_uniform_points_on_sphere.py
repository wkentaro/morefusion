import numpy as np

from .get_points_from_angles import get_points_from_angles


def get_uniform_points_on_sphere(n_sample, radius=1):
    elevation = np.linspace(-90, 90, n_sample)
    azimuth = np.linspace(-180, 180, n_sample, endpoint=False)
    elevation, azimuth = np.meshgrid(elevation, azimuth)

    # if elevation is -90 or 90, azimuth has no effect
    keep = elevation != -90
    keep[np.argmin(keep)] = True
    azimuth = azimuth[keep]
    elevation = elevation[keep]

    keep = elevation != 90
    keep[np.argmin(keep)] = True
    azimuth = azimuth[keep]
    elevation = elevation[keep]

    elevation = elevation.flatten()
    azimuth = azimuth.flatten()

    n_points = len(elevation)
    distance = np.full((n_points,), radius, dtype=float)
    points = get_points_from_angles(distance, elevation, azimuth)
    return points
