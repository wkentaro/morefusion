import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from objslampp.functions.geometry.voxelization_3d import voxelization_3d
from objslampp.functions.geometry.voxelization_3d import Voxelization3D


class TestVoxelization3D(unittest.TestCase):

    def setUp(self):
        voxel_dim = 32
        self.dimensions = (voxel_dim, voxel_dim, voxel_dim)
        self.channels = 4

        self.origin = numpy.array([-1, -1, -1], dtype=numpy.float32)
        self.pitch = numpy.float32(2. / voxel_dim)

        n_points = 128
        self.points = numpy.random.uniform(
            -1, 1, (n_points, 3)
        ).astype(numpy.float32)
        self.values = numpy.random.uniform(
            -1, 1, (n_points, self.channels)
        ).astype(numpy.float32)

        shape = (self.channels,) + self.dimensions
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, values_data, points_data):
        values = chainer.Variable(values_data)
        points = chainer.Variable(points_data)
        y = voxelization_3d(
            values,
            points,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
            channels=self.channels,
        )
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.values, self.points)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.values), cuda.to_gpu(self.points))

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        function = Voxelization3D(
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
            channels=self.channels,
        )

        # cpu
        values_cpu = chainer.Variable(self.values)
        points_cpu = chainer.Variable(self.points)
        y_cpu = function(values_cpu, points_cpu)
        counts_cpu = function.counts

        # gpu
        values_gpu = chainer.Variable(cuda.to_gpu(self.values))
        points_gpu = chainer.Variable(cuda.to_gpu(self.points))
        y_gpu = function(values_gpu, points_gpu)
        counts_gpu = function.counts

        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))
        testing.assert_allclose(
            counts_cpu, cuda.to_cpu(counts_gpu), atol=0, rtol=0
        )

    def check_backward(self, values_data, points_data, y_grad):
        gradient_check.check_backward(
            Voxelization3D(
                pitch=self.pitch,
                origin=self.origin,
                dimensions=self.dimensions,
                channels=self.channels,
            ),
            (values_data, points_data), y_grad, no_grads=[False, True],
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.values, self.points, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.values), cuda.to_gpu(self.points),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
