#!/usr/bin/env python

import numpy as np
import trimesh.transformations as tf

import objslampp


def main():
    models = objslampp.datasets.YCBVideoModels()
    points = models.get_pcd(class_id=2)

    quaternion_true = tf.random_quaternion()
    quaternion_pred = quaternion_true + [0.1, 0, 0, 0]

    transform_true = tf.quaternion_matrix(quaternion_true)
    transform_pred = tf.quaternion_matrix(quaternion_pred)

    # translation
    translation_true = np.random.uniform(-0.02, 0.02, (3,))
    translation_pred = np.random.uniform(-0.02, 0.02, (3,))
    transform_true[:3, 3] = translation_true
    transform_pred[:3, 3] = translation_pred

    for symmetric in [False, True]:
        add = objslampp.functions.loss.average_distance_l1(
            points,
            transform_true[None],
            transform_pred[None],
            symmetric=symmetric,
        )
        print(f'symmetric={symmetric}, add={float(add.array[0])}')


if __name__ == '__main__':
    main()
