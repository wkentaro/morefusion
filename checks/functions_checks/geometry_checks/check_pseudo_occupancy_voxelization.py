#!/usr/bin/env python

import chainer
import numpy as np
import trimesh

import imgviz

import objslampp


scene = trimesh.Scene()
scene.add_geometry(trimesh.creation.axis(0.005))

models = objslampp.datasets.YCBVideoModels()

scenes = {}

for name in ['nonsolid', 'solid']:
    if name == 'nonsolid':
        points = models.get_pcd(class_id=2)
    else:
        assert name == 'solid'
        points = models.get_solid_voxel(class_id=2).points

    points -= points.min(axis=0)

    points = objslampp.extra.open3d.voxel_down_sample(points, voxel_size=0.005)
    points = points.astype(np.float32)

    geom = trimesh.PointCloud(vertices=points)
    scene.add_geometry(geom)

    points = chainer.cuda.to_gpu(points)

    print('voxelization start')
    dim = 16
    pitch = max(geom.extents) / dim * 1.1
    origin = np.array((- pitch / 2,) * 3, dtype=float)
    print(f'pitch: {pitch}')
    print(f'dim: {dim}')
    grid = objslampp.functions.pseudo_occupancy_voxelization(
        points,
        pitch=pitch,
        origin=tuple(origin),        # definition origin is center
        dims=(dim, dim, dim),
        threshold=2,
    ).array
    grid = chainer.cuda.to_cpu(grid)
    print('voxelization done')
    print(grid.min(), grid.max())
    colors = imgviz.depth2rgb(grid.reshape(1, -1), min_value=0, max_value=1)
    colors = colors.reshape(dim, dim, dim, 3)
    colors = np.concatenate((colors, np.full((dim, dim, dim, 1), 127)), axis=3)

    voxel = trimesh.voxel.Voxel(
        matrix=grid,
        pitch=pitch,
        origin=origin,
    )
    geom = voxel.as_boxes()
    I, J, K = zip(*np.argwhere(grid))
    geom.visual.face_colors = colors[I, J, K].repeat(12, axis=0)
    scene.add_geometry(geom)

    scenes[name] = scene

objslampp.extra.trimesh.display_scenes(scenes)
