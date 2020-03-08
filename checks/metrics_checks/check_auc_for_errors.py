#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import trimesh.transformations as tf

import morefusion


def main():
    models = morefusion.datasets.YCBVideoModels()
    points = models.get_pcd(class_id=2)

    n_test = 1000
    transforms_true = []
    transforms_pred = []
    for i in range(n_test):
        quaternion_true = tf.random_quaternion()
        quaternion_pred = quaternion_true + [0.1, 0, 0, 0]

        translation_true = np.random.uniform(-0.02, 0.02, (3,))
        translation_pred = np.random.uniform(-0.02, 0.02, (3,))

        transform_true = tf.quaternion_matrix(quaternion_true)
        transform_true[:3, 3] = translation_true
        transform_pred = tf.quaternion_matrix(quaternion_pred)
        transform_pred[:3, 3] = translation_pred

        transforms_true.append(transform_true)
        transforms_pred.append(transform_pred)

    adds = morefusion.metrics.average_distance(
        [points] * n_test, transforms_true, transforms_pred
    )[0]
    max_distance = 0.1
    auc, x, y = morefusion.metrics.auc_for_errors(
        adds, max_threshold=max_distance, return_xy=True
    )
    print("auc:", auc)

    plt.plot(x, y)
    plt.xlabel("add threshold")
    plt.xlim(0, max_distance)
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    main()
