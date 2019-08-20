#!/usr/bin/env python

import numpy as np
import trimesh

import objslampp

import contrib


dataset = contrib.datasets.YCBVideoDataset('train')

frame = dataset.get_frame(0)
examples = dataset.get_example(0)

scene = trimesh.Scene()
for example in examples:
    example['class_id']
    example['origin']
    example['pitch']

    mapping = objslampp.geometry.VoxelMapping(
        origin=example['origin'],
        pitch=example['pitch'],
        voxel_dim=32,
        nchannel=3,
    )
    mask = ~np.isnan(example['pcd']).any(axis=2)
    mapping.add(example['pcd'][mask], example['rgb'][mask])
    scene.add_geometry(mapping.as_boxes())
    scene.add_geometry(mapping.as_bbox())
scene.camera.resolution = (640, 480)
K = frame['intrinsic_matrix']
scene.camera.focal = (K[0, 0], K[1, 1])
scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform()

scenes = {'rgb': frame['rgb'], 'scene': scene}
objslampp.extra.trimesh.display_scenes(scenes)
