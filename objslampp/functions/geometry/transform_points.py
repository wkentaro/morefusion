import chainer
from chainer.backends import cuda
import chainer.functions as F


def transform_points(points, transform):
    N = points.shape[0]
    assert points.shape == (N, 3)

    squeeze_axis0 = False
    if transform.ndim == 2:
        transform = transform[None]
        squeeze_axis0 = True

    M = transform.shape[0]
    assert transform.shape == (M, 4, 4)

    if isinstance(points, chainer.Variable):
        xp = cuda.get_array_module(points.array)
    else:
        xp = cuda.get_array_module(points)

    points = F.concat([points, xp.ones((N, 1), dtype=points.dtype)], axis=1)
    # Mx4x4 @ 4xN -> Mx4xN
    points = F.matmul(transform, points.T)
    points = points.transpose(0, 2, 1)[:, :, :3]  # MxNx3

    if squeeze_axis0:
        points = points[0, :, :]  # 1xNx3 -> Nx3
    return points
