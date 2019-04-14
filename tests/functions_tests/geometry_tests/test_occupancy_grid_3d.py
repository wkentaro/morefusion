import unittest

import chainer.gradient_check
from chainer.testing import condition
import numpy as np

from objslampp.functions.geometry.occupancy_grid_3d import occupancy_grid_3d


class TestOccupancyGrid3D(unittest.TestCase):

    def setUp(self):
        self.pitch = 1
        self.origin = (0, 0, 0)
        self.dimension = (5, 5, 5)
        self.points = np.array(
            [[0, 0.05, 0.1], [3.9, 3.95, 4]],
            dtype=np.float32
        )
        self.grad_matrix = np.random.uniform(
            -1, 1, self.dimension
        ).astype(np.float32)

    def test_forward_cpu(self):
        matrix = occupancy_grid_3d(
            self.points,
            pitch=self.pitch,
            origin=self.origin,
            dimension=self.dimension,
        )
        nonzero = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [4, 3, 4],
            [3, 4, 4],
            [4, 4, 4],
        ]
        matrix_bool = np.zeros(matrix.shape, dtype=bool)
        matrix_bool[tuple(zip(*nonzero))] = True
        np.testing.assert_allclose((matrix.array > 0), matrix_bool, rtol=0)

    def check_backward(self, points_data, grad_matrix):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_3d(
                x,
                pitch=self.pitch,
                origin=self.origin,
                dimension=self.dimension,
            ),
            points_data,
            grad_matrix,
        )

    @condition.retry(5)
    def test_backward_cpu(self):
        self.check_backward(self.points, self.grad_matrix)
