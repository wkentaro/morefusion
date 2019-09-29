from chainer.backends import cuda
import chainer.functions as F

from ..geometry import transform_points
from ... import extra as extra_module


def average_distance_l2(points, transform1, transform2):
    """Translation introduced pose_loss proposed in PoseCNN paper.

        Original pose_loss looks like below:

        .. math::

            m = |M|
            PLOSS(\\tilde{q}, q) =
                1 / 2m \\sum_{x \\in M} || R(\\tilde{q})x - R(q)x ||^2,

        where M is set of point_xyz, q~ and q are quaternion,
        R(q~) and R(q) are rotation matrix of ground truth and predicted,
        and m is size of the set M.

        If we introduce translation here, it will be:

        .. math::

            PLOSS2(\\tilde{T}, T) =
                1 / 2m \\sum_{x \\in M} || \\tilde{T}x - Tx ||^2.

    """
    assert points.shape == (points.shape[0], 3)
    assert transform1.shape == (transform1.shape[0], 4, 4)
    assert transform2.shape == (transform1.shape[0], 4, 4)

    points1 = transform_points(points, transform1)
    points2 = transform_points(points, transform2)

    return F.mean(F.sum((points1 - points2) ** 2, axis=2), axis=1) / 2.


def average_distance(points, transform_true, transforms_pred, symmetric=False):
    """Translation introduced pose_loss proposed in DenseFusion paper.

        Original pose_loss looks like below:

        .. math::

            m = |M|
            PLOSS(\\tilde{q}, q) =
                1 / m \\sum_{x \\in M} | R(\\tilde{q})x - R(q)x |

        where M is set of point_xyz, q~ and q are quaternion,
        R(q~) and R(q) are rotation matrix of ground truth and predicted,
        and m is size of the set M.

        If we introduce translation here, it will be:

        .. math::

            PLOSS2(\\tilde{T}, T) =
                1 / m \\sum_{x \\in M} | \\tilde{T}x - Tx |

    """
    n_points = points.shape[0]
    n_pred = transforms_pred.shape[0]
    assert points.shape == (n_points, 3)
    assert transform_true.shape == (4, 4)
    assert transforms_pred.shape == (n_pred, 4, 4)

    points_true = transform_points(points, transform_true)
    points_pred = transform_points(points, transforms_pred)
    assert points_true.shape == (n_points, 3)
    assert points_pred.shape == (n_pred, n_points, 3)

    if symmetric:
        points_true_array = cuda.to_cpu(points_true.array)
        points_pred_array = cuda.to_cpu(points_pred.array)
        points_pred_array = points_pred_array.reshape(n_pred * n_points, 3)

        ref = points_true_array.transpose(1, 0)[None]
        query = points_pred_array.transpose(1, 0)[None]
        indices = extra_module.knn_cuda.knn_cuda(1, ref, query)[0, 0]

        points_true = points_true[indices]
        points_true = points_true.reshape(n_pred, n_points, 3)
    else:
        points_true = F.repeat(points_true[None], n_pred, axis=0)

    return F.mean(
        F.sqrt(F.sum((points_true - points_pred) ** 2, axis=2)), axis=1
    )
