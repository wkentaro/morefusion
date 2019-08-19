#!/usr/bin/env python

import chainer
import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp

import contrib


dataset = contrib.datasets.YCBVideoDataset('train')

np.random.seed(0)
indices = np.random.permutation(len(dataset))

examples1 = dataset.get_example(indices[0])
examples2 = dataset.get_example(indices[1])
frame1 = dataset.get_frame(indices[0])
frame2 = dataset.get_frame(indices[1])

examples1 = chainer.dataset.concat_examples(examples1)
examples2 = chainer.dataset.concat_examples(examples2)

index1 = np.where(examples1['class_id'] == 12)[0][0]
pcd1 = examples1['pcd'][index1]
rgb1 = examples1['rgb'][index1]
mask1 = ~np.isnan(pcd1).any(axis=2)
index2 = np.where(examples2['class_id'] == 12)[0][0]
pcd2 = examples2['pcd'][index2]
rgb2 = examples2['rgb'][index2]
mask2 = ~np.isnan(pcd2).any(axis=2)

T_cad2cam1 = tf.quaternion_matrix(examples1['quaternion_true'][index1])
T_cad2cam1 = objslampp.geometry.compose_transform(
    T_cad2cam1[:3, :3], examples1['translation_true'][index1]
)
T_cad2cam2 = tf.quaternion_matrix(examples2['quaternion_true'][index2])
T_cad2cam2 = objslampp.geometry.compose_transform(
    T_cad2cam2[:3, :3], examples2['translation_true'][index2]
)


scene = trimesh.Scene()
geom = trimesh.PointCloud(vertices=pcd1[mask1], colors=rgb1[mask1])
scene.add_geometry(geom)
geom = trimesh.PointCloud(vertices=pcd2[mask2], colors=rgb2[mask2])
scene.add_geometry(geom, transform=T_cad2cam1 @ np.linalg.inv(T_cad2cam2))
scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform()
scene.camera.resolution = (320, 240)
scene.camera.focal = (frame1['intrinsic_matrix'][0, 0] / np.sqrt(2), frame1['intrinsic_matrix'][1, 1] / np.sqrt(2))

objslampp.extra.trimesh.display_scenes({
    'rgb1': frame1['rgb'],
    'rgb2': frame2['rgb'],
    'scene': scene,
}, tile=(1, 3), height=480 // 4 * 3, width=640 // 4 * 3)
