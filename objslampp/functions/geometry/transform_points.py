import chainer
from chainer.backends import cuda
import chainer.functions as F


def transform_points(points, transform):
    N = points.shape[0]
    assert points.shape == (N, 3)

    M = transform.shape[0]
    assert transform.shape == (M, 4, 4)

    if isinstance(points, chainer.Variable):
        xp = cuda.get_array_module(points.array)
    else:
        xp = cuda.get_array_module(points)

    points = F.concat([points, xp.ones((N, 1), dtype=points.dtype)], axis=1)
    # Mx4x4 @ 4xN -> Mx4xN
    points = F.matmul(transform, points.T)
    points = points.transpose(0, 2, 1)[:, :, :3]  # M, N, 3
    return points
