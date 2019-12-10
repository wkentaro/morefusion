import typing

import numpy as np


def project_to_camera(
    points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_shape: typing.Optional[tuple] = None,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    assert points.ndim == 2, 'points.ndim must be 2'
    if image_shape is not None:
        assert len(image_shape) in (2, 3), \
            'image_shape must be (H, W) or (H, W, C)'

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    c = cx + (x / z * fx)
    r = cy + (y / z * fy)

    if image_shape is not None:
        r = np.clip(r, 0, image_shape[0] - 1)
        c = np.clip(c, 0, image_shape[1] - 1)
    return r, c
