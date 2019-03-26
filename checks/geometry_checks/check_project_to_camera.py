#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import trimesh

import objslampp


dataset = objslampp.datasets.YCBVideoDataset('train', sampling=60)
view1 = dataset[0]
view2 = dataset[1]

K1 = view1['meta']['intrinsic_matrix']
color1 = view1['color']
depth1 = view1['depth']
# plt.imshow(color1)
# plt.show()

K2 = view2['meta']['intrinsic_matrix']
color2 = view2['color']
depth2 = view2['depth']
# plt.imshow(color2)
# plt.show()

# project pcd2 (view2/cam2 frame -> world frame)
pcd2_cam2 = objslampp.geometry.pointcloud_from_depth(
    depth2,
    fx=K2[0, 0],
    fy=K2[1, 1],
    cx=K2[0, 2],
    cy=K2[1, 2],
)
isnan = np.isnan(pcd2_cam2).any(axis=2)
pcd2_cam2 = pcd2_cam2[~isnan]
T2_world_to_cam = view2['meta']['rotation_translation_matrix']
T2_world_to_cam = np.r_[T2_world_to_cam, [[0, 0, 0, 1]]]
T2_cam_to_world = np.linalg.inv(T2_world_to_cam)
pcd2_world = trimesh.transform_points(pcd2_cam2, T2_cam_to_world)

# project pcd2 (world frame -> view1/cam1 frame)
T1_world_to_cam = view1['meta']['rotation_translation_matrix']
T1_world_to_cam = np.r_[T1_world_to_cam, [[0, 0, 0, 1]]]
pcd2_cam1 = trimesh.transform_points(pcd2_world, T1_world_to_cam)

# project to camera (view1/cam1 frame)
r, c = objslampp.geometry.project_to_camera(
    pcd2_cam1,
    fx=K1[0, 0],
    fy=K1[1, 1],
    cx=K1[0, 2],
    cy=K1[1, 2],
    image_shape=color1.shape,
)
r = r.round().astype(int)
c = c.round().astype(int)
color2_cam1 = np.zeros_like(color1)
color2_cam1[r, c] = color2[~isnan]
mask = np.zeros(color1.shape[:2], dtype=bool)
mask[r, c] = True
diff = np.full(color1.shape[:2], float('nan'), dtype=float)
diff[mask] = (np.abs(color1[mask] - color2_cam1[mask]) / 255.).mean(axis=1)

plt.subplot(131)
plt.imshow(color1)
plt.subplot(132)
plt.imshow(color2_cam1)
plt.subplot(133)
plt.imshow(diff, cmap='jet')
plt.show()
