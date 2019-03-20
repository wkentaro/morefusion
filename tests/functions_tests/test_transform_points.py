import unittest

from chainer import cuda
from chainer.testing import attr
import numpy as np
import trimesh

from objslampp import functions


class TestTransformPoints(unittest.TestCase):

    def setUp(self):
        n_points = 128
        self.points = np.random.uniform(
            -1, 1, (n_points, 3)
        ).astype(np.float32)

        transform = trimesh.transformations.random_rotation_matrix()
        transform[:3, 3] = np.random.uniform(1, 2, (3,))
        self.transform = transform.astype(np.float32)

    def check_forward(self, points, transform):
        points1 = trimesh.transform_points(
            cuda.to_cpu(self.points), cuda.to_cpu(self.transform)
        )
        points2 = functions.transform_points(
            self.points, self.transform
        ).array
        np.testing.assert_allclose(points1, points2, atol=1e-5, rtol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.points, self.transform)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.points), cuda.to_gpu(self.transform)
        )
