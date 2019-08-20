from .compose_transform import compose_transform
from .quaternion_matrix import quaternion_matrix


def transformation_matrix(quaternion, translation):
    if quaternion.ndim == 2:
        batch_size = quaternion.shape[0]
        assert quaternion.shape == (batch_size, 4)
        assert translation.shape == (batch_size, 3)
        T = quaternion_matrix(quaternion)
        T = compose_transform(T[:, :3, :3], translation)
    else:
        assert quaternion.ndim == 1
        assert quaternion.shape == (4,)
        assert translation.shape == (3,)
        T = quaternion_matrix(quaternion[None])[0]
        T = compose_transform(T[None, :3, :3], translation[None])[0]
    return T
