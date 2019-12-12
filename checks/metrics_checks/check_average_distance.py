#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion


def main():
    models = morefusion.datasets.YCBVideoModels()
    points = models.get_pcd(class_id=2)

    quaternion_true = tf.random_quaternion()
    quaternion_pred = quaternion_true + [0.1, 0, 0, 0]

    transform_true = tf.quaternion_matrix(quaternion_true)
    transform_pred = tf.quaternion_matrix(quaternion_pred)

    scenes = {}
    for use_translation in [False, True]:
        if use_translation:
            translation_true = np.random.uniform(-0.02, 0.02, (3,))
            translation_pred = np.random.uniform(-0.02, 0.02, (3,))
            transform_true[:3, 3] = translation_true
            transform_pred[:3, 3] = translation_pred

        add = morefusion.metrics.average_distance(
            [points], [transform_true], [transform_pred]
        )[0][0]

        # ---------------------------------------------------------------------

        scene = trimesh.Scene()

        points_true = tf.transform_points(points, transform_true)
        colors = np.full((points_true.shape[0], 3), [1., 0, 0])
        geom = trimesh.PointCloud(vertices=points_true, color=colors)
        scene.add_geometry(geom)

        points_pred = tf.transform_points(points, transform_pred)
        colors = np.full((points_true.shape[0], 3), [0, 0, 1.])
        geom = trimesh.PointCloud(vertices=points_pred, color=colors)
        scene.add_geometry(geom)

        scenes[f'use_translation: {use_translation}, add: {add}'] = scene
        if scenes:
            camera_transform = list(scenes.values())[0].camera_transform
            scene.camera_transform = camera_transform

    morefusion.extra.trimesh.display_scenes(scenes)


if __name__ == '__main__':
    main()
