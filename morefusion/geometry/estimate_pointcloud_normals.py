import numpy as np
import open3d


def estimate_pointcloud_normals(points):
    if points.ndim == 3:
        return _estimate_pointcloud_normals_organized(points)
    elif points.ndim == 2:
        return _estimate_pointcloud_normals_unorganized(points)
    else:
        raise ValueError("points shape must be either (H, W, 3) or (N, 3)")


def _estimate_pointcloud_normals_unorganized(points):
    assert points.shape[1] == 3

    nonnan = ~np.isnan(points).any(axis=1)
    points_open3d = open3d.PointCloud()
    points_open3d.points = open3d.Vector3dVector(points[nonnan])
    open3d.estimate_normals(
        points_open3d,
        search_param=open3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
    )
    return np.asarray(points_open3d.normals)


# Modified from https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_surface_normals.py  # NOQA
def _estimate_pointcloud_normals_organized(points):
    # These lookups denote yx offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    assert points.shape[2] == 3

    d = 2
    H, W = points.shape[:2]
    points = np.pad(
        points,
        pad_width=((d, d), (d, d), (0, 0)),
        mode="constant",
        constant_values=np.nan,
    )
    lookups = np.array(
        [(-d, 0), (-d, d), (0, d), (d, d), (d, 0), (d, -d), (0, -d), (-d, -d)]
    )

    j, i = np.meshgrid(np.arange(W), np.arange(H))
    k = np.arange(8)

    i1 = i + d
    j1 = j + d
    points1 = points[i1, j1]

    lookup = lookups[k]
    i2 = i1[None, :, :] + lookup[:, 0, None, None]
    j2 = j1[None, :, :] + lookup[:, 1, None, None]
    points2 = points[i2, j2]

    lookup = lookups[(k + 2) % 8]
    i3 = i1[None, :, :] + lookup[:, 0, None, None]
    j3 = j1[None, :, :] + lookup[:, 1, None, None]
    points3 = points[i3, j3]

    diff = np.linalg.norm(points2 - points1, axis=3) + np.linalg.norm(
        points3 - points1, axis=3
    )
    diff[np.isnan(diff)] = np.inf
    indices = np.argmin(diff, axis=0)

    normals = np.cross(
        points2[indices, i, j] - points1[i, j],
        points3[indices, i, j] - points1[i, j],
    )
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    return normals
