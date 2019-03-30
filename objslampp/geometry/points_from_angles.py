from chainer.backends import cuda


def points_from_angles(
    distance,
    elevation,
    azimuth,
    is_degree: bool = True,
):
    xp = cuda.get_array_module(distance)

    distance = xp.asarray(distance)
    elevation = xp.asarray(elevation)
    azimuth = xp.asarray(azimuth)

    assert distance.ndim == elevation.ndim == azimuth.ndim == 1
    assert len(distance) == len(elevation) == len(azimuth)

    if is_degree:
        elevation = xp.radians(elevation)
        azimuth = xp.radians(azimuth)
    return xp.stack([
        distance * xp.cos(elevation) * xp.sin(azimuth),
        - distance * xp.cos(elevation) * xp.cos(azimuth),
        distance * xp.sin(elevation),
    ]).transpose()
