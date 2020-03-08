import unittest

from chainer.backends import cuda
import chainer.gradient_check
from chainer import testing
from chainer.testing import condition
import numpy as np

from morefusion.functions.geometry.occupancy_grid_3d import occupancy_grid_3d


class TestOccupancyGrid3D(unittest.TestCase):
    def setUp(self):
        self.pitch = 1
        self.origin = (0, 0, 0)
        self.dims = (5, 5, 5)
        self.points = np.array(
            [[0, 0.05, 0.1], [3.9, 3.95, 4]], dtype=np.float32
        )
        self.grad_matrix = np.random.uniform(-1, 1, self.dims).astype(
            np.float32
        )

    def check_forward(self, points_data):
        matrix = occupancy_grid_3d(
            points_data, pitch=self.pitch, origin=self.origin, dims=self.dims,
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
        testing.assert_allclose(matrix.array > 0, matrix_bool, rtol=0, atol=0)

    def test_forward_cpu(self):
        self.check_forward(self.points)

    @testing.attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.points))

    @testing.attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        y_cpu = occupancy_grid_3d(
            self.points, pitch=self.pitch, origin=self.origin, dims=self.dims,
        )

        y_gpu = occupancy_grid_3d(
            cuda.to_gpu(self.points),
            pitch=self.pitch,
            origin=self.origin,
            dims=self.dims,
        )

        testing.assert_allclose(y_cpu.array, cuda.to_cpu(y_gpu.array))

    def check_backward(self, points_data, grad_matrix):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_3d(
                x, pitch=self.pitch, origin=self.origin, dims=self.dims,
            ),
            points_data,
            grad_matrix,
        )

    @condition.retry(5)
    def test_backward_cpu(self):
        self.check_backward(self.points, self.grad_matrix)

    @testing.attr.gpu
    @condition.retry(5)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.points), cuda.to_gpu(self.grad_matrix)
        )
