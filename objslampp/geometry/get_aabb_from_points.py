import numpy as np
import open3d
import scipy.cluster.hierarchy


def voxel_down_sample(points, voxel_size):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
    dst_points = np.asarray(pcd.points)
    return dst_points


def get_aabb_from_points(points):
    # getting aabb
    pcd_roi_flat_down = voxel_down_sample(
        points=points, voxel_size=0.01
    )
    fclusterdata = scipy.cluster.hierarchy.fclusterdata(
        pcd_roi_flat_down, criterion='distance', t=0.02
    )
    cluster_ids, cluster_counts = np.unique(
        fclusterdata, return_counts=True
    )
    cluster_id = cluster_ids[np.argmax(cluster_counts)]
    keep = fclusterdata == cluster_id
    pcd_roi_flat_down = pcd_roi_flat_down[keep]
    aabb_min = pcd_roi_flat_down.min(axis=0)
    aabb_max = pcd_roi_flat_down.max(axis=0)
    return aabb_min, aabb_max
