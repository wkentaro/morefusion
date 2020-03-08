import numpy as np
import scipy.spatial


def sort(points):
    assert points.ndim == 2, "points must be 2 dimensional"
    assert points.shape[1] == 3, "points shape must be (N, 3)"

    points_left = points.copy()[1:]
    points_sorted = [points[0]]
    while len(points_sorted) < (len(points) - 1):  # drop last point
        kdtree = scipy.spatial.KDTree(points_left)
        _, index = kdtree.query(points_sorted[-1])
        points_sorted.append(kdtree.data[index])
        points_left = points_left[np.arange(len(points_left)) != index]
    points_sorted = np.array(points_sorted, dtype=float)
    return points_sorted


def sort_by(points, key):
    assert points.ndim == 2, "points must be 2 dimensional"
    assert points.shape[1] == 3, "points shape must be (N, 3)"
    assert key.ndim == 2, "key must be 2 dimensional"
    assert key.shape[1] == 3, "key shape must be (N, 3)"
    assert len(points) == len(key), "points and key must be same size"

    points_sorted = []
    points_left = points.copy()
    for key_i in key:
        kdtree = scipy.spatial.KDTree(points_left)
        _, index = kdtree.query(key_i)
        points_sorted.append(points_left[index])
        points_left = points_left[np.arange(len(points_left)) != index]
    points_sorted = np.array(points_sorted)
    return points_sorted


def interpolate(keypoints, n_points):
    tick, _ = scipy.interpolate.splprep(keypoints.T, s=0)
    points = scipy.interpolate.splev(np.linspace(0, 1, n_points), tick)
    points = np.array(points, dtype=np.float64).T
    return points


# def random(xyz_min, xyz_max, n_keypoints=12, n_points=None):
#     if n_points is None:
#         n_points = n_keypoints * 6
#
#     keypoints = np.random.uniform(xyz_min, xyz_max, (n_keypoints, 3))
#     keypoints = sort(keypoints)
#
#     points = interpolate(keypoints, n_points)
#     return points
