import numpy as np
import trimesh.transformations as tf

from ..geometry import matrix_from_points


def _iou_3d(points, transform1, transform2, pitch):
    assert points.shape == (points.shape[0], 3)
    assert transform1.shape == (4, 4)
    assert transform2.shape == (4, 4)
    points1 = tf.transform_points(points, transform1)
    points2 = tf.transform_points(points, transform2)

    aabb_min = np.minimum(np.min(points1, axis=0), np.min(points2, axis=0))
    aabb_max = np.maximum(np.max(points1, axis=0), np.max(points2, axis=0))
    dims = np.ceil((aabb_max - aabb_min) / pitch).astype(int)

    matrix1 = matrix_from_points(
        points1, pitch=pitch, origin=aabb_min, dims=dims
    )
    matrix2 = matrix_from_points(
        points2, pitch=pitch, origin=aabb_min, dims=dims
    )

    import trimesh
    geom1 = trimesh.voxel.Voxel(matrix1, pitch=pitch, origin=aabb_min).as_boxes(colors=(1., 0, 0))
    geom2 = trimesh.voxel.Voxel(matrix2, pitch=pitch, origin=aabb_min).as_boxes(colors=(0., 1., 0))
    trimesh.Scene([geom1, geom2]).show()

    intersection = matrix1 & matrix2
    union = matrix1 | matrix2
    iou = intersection.sum() / union.sum()

    return iou


def iou_3d(points, transform1, transform2, pitch):
    assert isinstance(points, list)

    batch_size = len(points)
    assert len(transform1) == batch_size
    assert len(transform2) == batch_size

    ious = np.zeros((batch_size,), dtype=float)
    for i in range(batch_size):
        ious[i] = _iou_3d(points[i], transform1[i], transform2[i], pitch)
    return ious
