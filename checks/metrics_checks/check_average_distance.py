#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf
import pyglet

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()
    model = models.get_model(class_id=2)
    points = np.loadtxt(model['points_xyz'])

    quaternion_true = tf.random_quaternion()
    quaternion_pred = quaternion_true + [0.1, 0, 0, 0]

    transform_true = tf.quaternion_matrix(quaternion_true)
    transform_pred = tf.quaternion_matrix(quaternion_pred)

    for use_translation in [False, True]:
        if use_translation:
            translation_true = np.random.uniform(-0.02, 0.02, (3,))
            translation_pred = np.random.uniform(-0.02, 0.02, (3,))
            transform_true[:3, 3] = translation_true
            transform_pred[:3, 3] = translation_pred

        add = objslampp.metrics.average_distance(
            [points], [transform_true], [transform_pred]
        )[0]

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

        objslampp.extra.trimesh.show_with_rotation(
            scene,
            caption=f'use_translation: {use_translation}, add: {add}',
            start_loop=False,
        )

    pyglet.app.run()


if __name__ == '__main__':
    main()
