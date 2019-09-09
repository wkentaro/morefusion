import unittest

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import numpy as np

from objslampp.functions.geometry.interpolate_voxel_grid \
    import interpolate_voxel_grid
from objslampp.functions.geometry.interpolate_voxel_grid \
    import InterpolateVoxelGrid


class TestInterpolateVoxelGrid(unittest.TestCase):

    def setUp(self):
        n_channel = 4
        dim = 32
        n_point = 10

        self.voxelized = np.random.uniform(
            -1, 1, (n_channel, dim, dim, dim)
        ).astype(np.float32)
        self.indices = np.random.uniform(
            0, 31, (n_point, 3)
        ).astype(np.float32)
        self.gy = np.random.uniform(
            -1, 1, (n_point, n_channel)
        ).astype(np.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, voxelized_data, indices_data):
        voxelized = chainer.Variable(voxelized_data)
        indices = chainer.Variable(indices_data)
        y = interpolate_voxel_grid(voxelized, indices)
        self.assertEqual(y.data.dtype, np.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    # @condition.retry(3)
    # def test_forward_cpu(self):
    #     self.check_forward(self.values, self.points)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.voxelized), cuda.to_gpu(self.indices)
        )

    # @attr.gpu
    # @condition.retry(3)
    # def test_forward_cpu_gpu_equal(self):
    #     function = AverageVoxelization3D(
    #         origin=self.origin,
    #         pitch=self.pitch,
    #         dimensions=self.dimensions,
    #         channels=self.channels,
    #     )
    #
    #     # cpu
    #     values_cpu = chainer.Variable(self.values)
    #     points_cpu = chainer.Variable(self.points)
    #     y_cpu = function(values_cpu, points_cpu)
    #     counts_cpu = function.counts
    #
    #     # gpu
    #     values_gpu = chainer.Variable(cuda.to_gpu(self.values))
    #     points_gpu = chainer.Variable(cuda.to_gpu(self.points))
    #     y_gpu = function(values_gpu, points_gpu)
    #     counts_gpu = function.counts
    #
    #     testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))
    #     testing.assert_allclose(
    #         counts_cpu, cuda.to_cpu(counts_gpu), atol=0, rtol=0
    #     )

    def check_backward(self, voxelized_data, indices_data, y_grad):
        gradient_check.check_backward(
            InterpolateVoxelGrid(),
            (voxelized_data, indices_data), y_grad, no_grads=[False, True],
            **self.check_backward_options)

    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     self.check_backward(self.voxelized, self.indices, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.voxelized),
                            cuda.to_gpu(self.indices),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
