import chainer
from chainer.backends import cuda
import chainer.functions as F


def transform_points(points, transform):
    N = points.shape[0]
    assert points.shape == (N, 3)
    assert transform.shape == (4, 4)

    if isinstance(points, chainer.Variable):
        xp = cuda.get_array_module(points.array)
    else:
        xp = cuda.get_array_module(points)

    points = F.concat([points, xp.ones((N, 1), dtype=points.dtype)], axis=1)
    points = F.matmul(transform, points.T).T[:, :3]
    return points
