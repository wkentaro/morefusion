#!/usr/bin/env python

import numpy as np
import trimesh

import imgviz

import objslampp


scene = trimesh.Scene()
scene.add_geometry(trimesh.creation.axis(0.005))

models = objslampp.datasets.YCBVideoModels()
points = models.get_pcd(class_id=2).astype(np.float32)
points -= points.min(axis=0)

geom = trimesh.PointCloud(vertices=points)
scene.add_geometry(geom)

print('voxelization start')
dim = 16
pitch = max(geom.extents) / dim * 1.1
origin = np.array((- pitch / 2,) * 3, dtype=float)
print(f'pitch: {pitch}')
print(f'dim: {dim}')
grid = objslampp.functions.occupancy_grid_3d(
    points,
    pitch=pitch,
    origin=tuple(origin),        # definition origin is center
    dimension=(dim, dim, dim),
    threshold=2,
).array
print('voxelization done')
colors = imgviz.depth2rgb(grid.reshape(1, -1), min_value=0, max_value=1)
colors = colors.reshape(dim, dim, dim, 3)
colors = np.concatenate((colors, np.full((dim, dim, dim, 1), 127)), axis=3)

voxel = trimesh.voxel.Voxel(
    matrix=grid,
    pitch=pitch,
    origin=origin + pitch / 2,  # definition origin is right top corner
)
geom = voxel.as_boxes()
I, J, K = zip(*np.argwhere(grid))
geom.visual.face_colors = colors[I, J, K].repeat(12, axis=0)
scene.add_geometry(geom)

objslampp.extra.trimesh.show_with_rotation(scene, resolution=(500, 500))
