import numpy as np
import open3d


def voxel_down_sample(points, voxel_size):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
    return np.asarray(pcd.points)
