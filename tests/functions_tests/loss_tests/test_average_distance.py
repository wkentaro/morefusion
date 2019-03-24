import unittest

import trimesh.transformations as tf

from chainer.backends import cuda
from chainer import testing
import numpy as np

from objslampp import metrics
from objslampp import geometry
from objslampp.functions.loss.average_distance import average_distance


class TestAverageDistance(unittest.TestCase):

    def setUp(self):
        self.points = tf.random_vector((128, 3)).astype(np.float32)
        self.R1 = tf.random_rotation_matrix().astype(np.float32)
        self.t1 = tf.random_vector((3,)).astype(np.float32)
        self.R2 = tf.random_rotation_matrix().astype(np.float32)
        self.t2 = tf.random_vector((3,)).astype(np.float32)

    def check_forward_consistency(self, points, R1, R2, t1, t2, sqrt):
        T1 = geometry.compose_transform(R=R1[:3, :3], t=t1)
        T2 = geometry.compose_transform(R=R2[:3, :3], t=t2)
        d1 = average_distance(points, T1, T2, sqrt=sqrt)
        d1 = float(d1.array)
        d2 = average_distance(points, R1, R2, t1, t2, sqrt=sqrt)
        d2 = float(d2.array)
        np.testing.assert_allclose(d1, d2)

    def check_forward(self, points, R1, R2, t1, t2):
        d1 = average_distance(points, R1, R2, sqrt=True)
        d1 = float(d1.array)
        d2 = metrics.average_distance(
            [cuda.to_cpu(points)], [cuda.to_cpu(R1)], [cuda.to_cpu(R2)]
        )[0]
        np.testing.assert_allclose(d1, d2)

        T1 = geometry.compose_transform(R=R1[:3, :3], t=t1)
        T2 = geometry.compose_transform(R=R2[:3, :3], t=t2)
        d1 = average_distance(points, T1, T2, sqrt=True)
        d1 = float(d1.array)
        d2 = metrics.average_distance(
            [cuda.to_cpu(points)], [cuda.to_cpu(T1)], [cuda.to_cpu(T2)]
        )[0]
        np.testing.assert_allclose(d1, d2)

        self.check_forward_consistency(points, R1, R2, t1, t2, sqrt=False)
        self.check_forward_consistency(points, R1, R2, t1, t2, sqrt=True)

    def test_forward_cpu(self):
        self.check_forward(
            self.points,
            self.R1,
            self.R2,
            self.t1,
            self.t2,
        )

    @testing.attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.R1),
            cuda.to_gpu(self.R2),
            cuda.to_gpu(self.t1),
            cuda.to_gpu(self.t2),
        )
