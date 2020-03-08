from chainer.backends import cuda


def points_from_angles(
    distance, elevation, azimuth, is_degree: bool = True,
):
    xp = cuda.get_array_module(distance)

    distance = xp.asarray(distance)
    elevation = xp.asarray(elevation)
    azimuth = xp.asarray(azimuth)
    if is_degree:
        elevation = xp.radians(elevation)
        azimuth = xp.radians(azimuth)

    assert distance.shape == elevation.shape == azimuth.shape
    assert distance.ndim in (0, 1)

    return xp.stack(
        [
            distance * xp.cos(elevation) * xp.sin(azimuth),
            -distance * xp.cos(elevation) * xp.cos(azimuth),
            distance * xp.sin(elevation),
        ]
    ).transpose()
