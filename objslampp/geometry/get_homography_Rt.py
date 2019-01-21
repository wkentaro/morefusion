import typing

import numpy as np


def get_homography_Rt(
    R: typing.Optional[np.ndarray] = None,
    t: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    transform = np.eye(4)
    if R is not None:
        assert R.shape == (3, 3), 'rotation matrix R must be (3, 3) float'
        transform[:3, :3] = R
    if t is not None:
        assert t.shape == (3,), 'translation vector t must be (3,) float'
        transform[:3, 3] = t
    return transform
