import chainer.functions as F

from ..transform_points import transform_points


def average_distance(
    points,
    transform1,
    transform2,
    translation1=None,
    translation2=None,
):
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
    assert transform1.shape == (4, 4)
    assert transform2.shape == (4, 4)

    points1 = transform_points(points, transform1)
    if translation1 is not None:
        points1 = points1 + translation1

    points2 = transform_points(points, transform2)
    if translation2 is not None:
        points2 = points2 + translation2

    return F.mean(F.sum((points1 - points2) ** 2, axis=1)) / 2.
