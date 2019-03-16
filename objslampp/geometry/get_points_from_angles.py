import typing

from chainer.backends import cuda
import numpy as np


def get_points_from_angles(
    distance: typing.Any,
    elevation: typing.Any,
    azimuth: typing.Any,
    degrees: bool = True,
):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = np.radians(elevation)
            azimuth = np.radians(azimuth)
        return (
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
            -distance * np.cos(elevation) * np.cos(azimuth),
        )
    else:
        xp = cuda.get_array_module(distance)
        if degrees:
            elevation = xp.radians(elevation)
            azimuth = xp.radians(azimuth)
        return xp.stack([
            distance * xp.cos(elevation) * xp.sin(azimuth),
            distance * xp.sin(elevation),
            -distance * xp.cos(elevation) * xp.cos(azimuth),
        ]).transpose()
