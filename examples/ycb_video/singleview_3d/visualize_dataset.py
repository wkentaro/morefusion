#!/usr/bin/env python

import argparse

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion

import contrib


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--randomize-base', action='store_true')
args = parser.parse_args()

models = morefusion.datasets.YCBVideoModels()
dataset = contrib.datasets.YCBVideoDataset('train')

frame = dataset.get_frame(0)
examples = dataset.get_example(0)

scene = trimesh.Scene()
for example in examples:
    mask = ~np.isnan(example['pcd']).any(axis=2)
    points = example['pcd'][mask]
    values = example['rgb'][mask]

    T_cad2cam = morefusion.functions.transformation_matrix(
        example['quaternion_true'], example['translation_true']
    ).array
    if args.randomize_base:
        points = tf.transform_points(points, np.linalg.inv(T_cad2cam))
        T_rand = tf.random_rotation_matrix()
        T_cad2cam = T_cad2cam @ T_rand
        points = tf.transform_points(points, T_cad2cam)

    pitch = models.get_voxel_pitch(32, example['class_id'])
    center = np.median(points, axis=0)
    origin = center - pitch * 15.5

    mapping = morefusion.geometry.VoxelMapping(
        origin=origin,
        pitch=pitch,
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
scene.camera.transform = morefusion.extra.trimesh.to_opengl_transform()

scenes = {'rgb': frame['rgb'], 'scene': scene}
morefusion.extra.trimesh.display_scenes(scenes)
