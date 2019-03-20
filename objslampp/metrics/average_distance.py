import numpy as np
import trimesh.transformations as tf


def _average_distance(points, transform1, transform2):
    assert points.shape == (points.shape[0], 3)
    assert transform1.shape == (4, 4)
    assert transform2.shape == (4, 4)
    points1 = tf.transform_points(points, transform1)
    points2 = tf.transform_points(points, transform2)
    return np.linalg.norm(points1 - points2, axis=1).mean()


def average_distance(points, transform1, transform2):
    batch_size = len(points)
    assert len(transform1) == batch_size
    assert len(transform2) == batch_size
    adds = np.zeros((batch_size,), dtype=float)
    for i in range(batch_size):
        adds[i] = _average_distance(points[i], transform1[i], transform2[i])
    return adds
