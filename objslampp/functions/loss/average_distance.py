from chainer.backends import cuda
import chainer.functions as F
import sklearn.neighbors

from ..geometry import transform_points


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


def average_distance_l1(
    points, transform1, transform2, symmetric=False, ohem_threshold=0,
):
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
    xp = cuda.get_array_module(points)

    assert points.shape == (points.shape[0], 3)
    batch_size = transform1.shape[0]
    assert transform1.shape == (batch_size, 4, 4)
    assert transform2.shape == (batch_size, 4, 4)

    points1 = transform_points(points, transform1)
    points2 = transform_points(points, transform2)

    if symmetric:
        n_hard_example = []
        points2_match = []
        for i in range(batch_size):
            points1_array = cuda.to_cpu(points1[i].array)
            points2_array = cuda.to_cpu(points2[i].array)
            kdtree = sklearn.neighbors.KDTree(points2_array)
            dists, indices = kdtree.query(points1_array)
            dists, indices = dists[:, 0], indices[:, 0]
            if ohem_threshold == 0:
                n_hard_example.append(len(dists))
            else:
                n = (dists >= ohem_threshold).sum()
                n_hard_example.append(1 if n == 0 else n)
            points2_match.append(points2[i][indices])
        n_hard_example = xp.array(n_hard_example)
        points2_match = F.concat(points2_match, axis=0)
        points2 = points2_match

    distances = F.sqrt(F.sum((points1 - points2) ** 2, axis=2))
    if ohem_threshold == 0:
        return F.mean(distances, axis=1)
    else:
        return F.sum(distances, axis=1) / n_hard_example
