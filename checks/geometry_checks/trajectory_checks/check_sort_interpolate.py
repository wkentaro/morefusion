#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp


aabb_min = (-1, -1, -1)
aabb_max = (1, 1, 1)

keypoints = np.random.uniform(aabb_min, aabb_max, (16, 3))
keypoints = objslampp.geometry.trajectory.sort(keypoints)

trajectory = objslampp.geometry.trajectory.interpolate(keypoints, n_points=128)

# -----------------------------------------------------------------------------

scene = trimesh.Scene()

box = trimesh.path.creation.box_outline((2, 2, 2))
scene.add_geometry(box)

axis = trimesh.creation.axis(0.01, origin_color=(0, 0, 0))
for point in keypoints:
    transform = tf.translation_matrix(point)
    scene.add_geometry(axis, transform=transform)

axis = trimesh.creation.axis(0.01)
for point in trajectory:
    transform = tf.translation_matrix(point)
    scene.add_geometry(axis, transform=transform)

objslampp.extra.trimesh.show_with_rotation(scene)
