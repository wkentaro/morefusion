#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion


n_keypoints = 16
n_points = 128

# target
aabb_min_target = (-0.2, -0.2, -0.2)
aabb_max_target = (0.2, 0.2, 0.2)
targets = np.random.uniform(
    aabb_min_target, aabb_max_target, (n_keypoints, 3)
)
targets = morefusion.geometry.trajectory.sort(targets)
targets = morefusion.geometry.trajectory.interpolate(targets, n_points=n_points)

# eye
aabb_min_eye = (-1, -1, -1)
aabb_max_eye = (1, 1, 1)
distance = np.full((n_keypoints,), 1, dtype=float)
elevation = np.random.uniform(30, 90, (n_keypoints,))
azimuth = np.random.uniform(0, 360, (n_keypoints,))
eyes = morefusion.geometry.points_from_angles(distance, elevation, azimuth)
indices = indices = np.linspace(0, 127, num=len(eyes))
indices = indices.round().astype(int)
eyes = morefusion.geometry.trajectory.sort_by(eyes, key=targets[indices])
eyes = morefusion.geometry.trajectory.interpolate(eyes, n_points=128)

# -----------------------------------------------------------------------------

scene = trimesh.Scene()

box = trimesh.path.creation.box_outline((2, 2, 2))
scene.add_geometry(box)

axis = trimesh.creation.axis(0.01)
point = trimesh.creation.icosphere(radius=0.01, color=(1., 0, 0))
for eye, target in zip(eyes, targets):
    transform = tf.translation_matrix(eye)
    scene.add_geometry(axis, transform=transform)

    transform = tf.translation_matrix(target)
    scene.add_geometry(point, transform=transform)

    ray = trimesh.load_path([eye, target])
    scene.add_geometry(ray)

morefusion.extra.trimesh.show_with_rotation(scene, resolution=(400, 400))
