import unittest

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy as np

from morefusion.functions.geometry.interpolate_voxel_grid \
    import interpolate_voxel_grid
from morefusion.functions.geometry.interpolate_voxel_grid \
    import InterpolateVoxelGrid


class TestInterpolateVoxelGrid(unittest.TestCase):

    def setUp(self):
        batch_size = 3
        n_channel = 4
        dim = 32
        n_point = 128

        self.voxelized = np.random.uniform(
            -1, 1, (batch_size, n_channel, dim, dim, dim)
        ).astype(np.float32)
        self.points = np.random.uniform(
            0, 31, (n_point, 3)
        ).astype(np.float32)
        self.batch_indices = np.random.randint(
            0, batch_size, size=self.points.shape[0], dtype=np.int32
        )
        self.gy = np.random.uniform(
            -1, 1, (n_point, n_channel)
        ).astype(np.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, voxelized_data, points_data, batch_indices_data):
        y = interpolate_voxel_grid(
            voxelized_data, points_data, batch_indices_data
        )
        self.assertEqual(y.data.dtype, np.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.voxelized, self.points, self.batch_indices)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.voxelized),
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.batch_indices),
        )

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        y_cpu = interpolate_voxel_grid(
            self.voxelized, self.points, self.batch_indices
        )
        y_gpu = interpolate_voxel_grid(
            cuda.to_gpu(self.voxelized),
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.batch_indices),
        )
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(
        self, voxelized_data, points_data, batch_indices_data, y_grad
    ):
        gradient_check.check_backward(
            InterpolateVoxelGrid(),
            (voxelized_data, points_data, batch_indices_data),
            y_grad,
            no_grads=[False, True, True],
            **self.check_backward_options,
        )

    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     self.check_backward(
    #         self.voxelized, self.points, self.batch_indices, self.gy
    #     )

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.voxelized),
                            cuda.to_gpu(self.points),
                            cuda.to_gpu(self.batch_indices),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
