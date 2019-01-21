import typing

import numpy as np

from .get_homography_Rt import get_homography_Rt


def normalize(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 1, 'x must be a vector (ndim: 1)'
    return x / np.linalg.norm(x)


def look_at(
    eye,
    at: typing.Optional[typing.Any] = None,
    up: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.

    Parameters
    ----------
    eye: (3,) float
        Camera position.
    at: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).

    Returns
    -------
    R, t: (3, 3) float, (3,) float (if return_homography is False)
        Rotation matrix and translation vector.
        Points are transformed like below:
            y = (x - t) @ R.T  (transform points world -> camera)
            x = (y @ R) + t    (transform points camera -> world)

    T_cam2world: (4, 4) float (if return_homography is True)
        Homography transformation matrix from camera to world.
        Points are transformed like below:
            y = trimesh.transforms.transform_points(x, T_cam2world)
            x = trimesh.transforms.transform_points(
                y, np.linalg.inv(T_cam2world)
            )
    """
    eye = np.asarray(eye, dtype=float)

    if at is None:
        at = np.array([0, 0, 0], dtype=float)
    else:
        at = np.asarray(at, dtype=float)

    if up is None:
        up = np.array([0, 1, 0], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), 'eye must be (3,) float'
    assert at.shape == (3,), 'at must be (3,) float'
    assert up.shape == (3,), 'at must be (3,) float'

    # create new axes
    z_axis = normalize(at - eye)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R = np.vstack((x_axis, y_axis, z_axis))
    t = eye

    T_cam2world = get_homography_Rt(R=R.T, t=t)
    return T_cam2world
