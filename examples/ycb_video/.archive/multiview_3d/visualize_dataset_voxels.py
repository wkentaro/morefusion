#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp

import contrib


models = objslampp.datasets.YCBVideoModels()
dataset = contrib.datasets.YCBVideoDataset('train')

frame = dataset.get_frame(0)
examples = dataset.get_example(0)

scene = trimesh.Scene()
for example in examples:
    mask = ~np.isnan(example['pcd']).any(axis=2)
    points = example['pcd'][mask]
    values = example['rgb'][mask]

    T_cad2cam = objslampp.functions.transformation_matrix(
        example['quaternion_true'], example['translation_true']
    ).array
    T_cam2cad = np.linalg.inv(T_cad2cam)
    T_random_rot = tf.random_rotation_matrix()
    points = tf.transform_points(points, T_cam2cad)
    points = tf.transform_points(points, T_random_rot)
    points = tf.transform_points(points, T_cad2cam)
    T_cad2cam = T_cad2cam @ T_random_rot

    mapping = objslampp.geometry.VoxelMapping(
        origin=example['origin'],
        pitch=example['pitch'],
        voxel_dim=32,
        nchannel=3,
    )
    mapping.add(points, values)
    scene.add_geometry(mapping.as_boxes())
    scene.add_geometry(mapping.as_bbox())

    cad = models.get_cad(example['class_id'])
    if hasattr(cad.visual, 'to_color'):
        cad.visual = cad.visual.to_color()
    scene.add_geometry(cad, transform=T_cad2cam)
scene.camera.resolution = (640, 480)
K = frame['intrinsic_matrix']
scene.camera.focal = (K[0, 0], K[1, 1])
scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform()

scenes = {'rgb': frame['rgb'], 'scene': scene}
objslampp.extra.trimesh.display_scenes(scenes)
