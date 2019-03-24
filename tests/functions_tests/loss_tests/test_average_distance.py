import unittest

import trimesh.transformations as tf

from chainer.backends import cuda
from chainer import testing
import numpy as np

from objslampp import metrics
from objslampp.functions.loss.average_distance import average_distance


class TestAverageDistance(unittest.TestCase):

    def setUp(self):
        self.points = tf.random_vector((128, 3)).astype(np.float32)
        self.T1 = tf.random_rotation_matrix().astype(np.float32)
        self.T1[:3, 3] = tf.random_vector((3,))
        self.T2 = tf.random_rotation_matrix().astype(np.float32)
        self.T2[:3, 3] = tf.random_vector((3,))

    def check_forward(self, points, T1, T2):
        d1 = average_distance(points, T1, T2, sqrt=True)
        d1 = float(d1.array)
        d2 = metrics.average_distance(
            [cuda.to_cpu(points)], [cuda.to_cpu(T1)], [cuda.to_cpu(T2)]
        )[0]
        testing.assert_allclose(d1, d2)

    def test_forward_cpu(self):
        self.check_forward(self.points, self.T1, self.T2)

    @testing.attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.T1),
            cuda.to_gpu(self.T2),
        )
