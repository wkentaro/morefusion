import numpy as np
import trimesh.transformations as tf
import sklearn.neighbors


def _average_distance(points, transform1, transform2, translate=True):
    assert points.shape == (points.shape[0], 3)
    assert transform1.shape == (4, 4)
    assert transform2.shape == (4, 4)
    points1 = tf.transform_points(points, transform1, translate=translate)
    points2 = tf.transform_points(points, transform2, translate=translate)

    add = np.linalg.norm(points1 - points2, axis=1).mean()

    kdtree = sklearn.neighbors.KDTree(points2)
    indices = kdtree.query(points1, return_distance=False)[:, 0]
    add_s = np.linalg.norm(points1 - points2[indices], axis=1).mean()

    return add, add_s


def average_distance(points, transform1, transform2, translate=True):
    assert isinstance(points, list)

    batch_size = len(points)
    assert len(transform1) == batch_size
    assert len(transform2) == batch_size

    adds = np.zeros((batch_size,), dtype=float)
    add_ss = np.zeros((batch_size,), dtype=float)
    for i in range(batch_size):
        adds[i], add_ss[i] = _average_distance(
            points[i], transform1[i], transform2[i], translate=translate
        )
    return adds, add_ss
