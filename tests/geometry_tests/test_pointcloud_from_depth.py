import numpy as np
import trimesh

from objslampp.geometry import pointcloud_from_depth


def test_pointcloud_from_depth():
    H, W = 256, 256
    K = trimesh.scene.Camera(resolution=(W, H), fov=(60, 60)).K

    depth = np.random.uniform(0, 3, (H, W))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pcd = pointcloud_from_depth(depth, fx, fy, cx, cy)
    assert pcd.shape == (H, W, 3)
    assert pcd.dtype == np.float64
