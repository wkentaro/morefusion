#!/usr/bin/env python

from chainer import cuda
import numpy as np
import trimesh

import imgviz

import objslampp


scene = trimesh.Scene()
scene.add_geometry(trimesh.creation.axis(0.005))

models = objslampp.datasets.YCBVideoModels()

scenes = {}

for is_solid in ['nonsolid', 'solid']:
    name = f'{is_solid}'

    if is_solid == 'nonsolid':
        points = models.get_pcd(class_id=2)
    else:
        assert is_solid == 'solid'
        points = models.get_solid_voxel(class_id=2).points

    points = objslampp.extra.open3d.voxel_down_sample(
        points, voxel_size=0.01
    )
    points = points.astype(np.float32)

    geom = trimesh.PointCloud(vertices=points)
    scene.add_geometry(geom)

    dim = 16
    pitch = max(geom.extents) / dim * 1.1
    origin = (- pitch * dim / 2,) * 3
    sdf = models.get_cad(class_id=2).nearest.signed_distance(points)
    print(f'[{name}] pitch: {pitch}')
    print(f'[{name}] dim: {dim}')
    grid = objslampp.functions.pseudo_occupancy_voxelization(
        points=cuda.to_gpu(points),
        sdf=cuda.to_gpu(sdf),
        pitch=pitch,
        origin=origin,
        dims=(dim,) * 3,
        threshold=2,
    ).array
    grid = cuda.to_cpu(grid)
    colors = imgviz.depth2rgb(
        grid.reshape(1, -1), min_value=0, max_value=1
    )
    colors = colors.reshape(dim, dim, dim, 3)
    colors = np.concatenate(
        (colors, np.full((dim, dim, dim, 1), 127)), axis=3
    )

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
