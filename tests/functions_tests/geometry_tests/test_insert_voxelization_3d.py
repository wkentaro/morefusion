'''
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from morefusion.functions.geometry.insert_voxelization_3d import \
    insert_voxelization_3d
from morefusion.functions.geometry.insert_voxelization_3d import \
    InsertVoxelization3D


class TestInsertVoxelization3D(unittest.TestCase):

    def setUp(self):
        voxel_dim = 32
        self.dimensions = (voxel_dim, voxel_dim, voxel_dim)
        self.channels = 4

        self.origin = numpy.array([-0.5, -0.5, -0.5], dtype=numpy.float32)
        self.pitch = numpy.float32(1. / voxel_dim)

        n_points = 128
        self.points = numpy.random.uniform(
            -0.5, 0.5, (n_points, 3)
        ).astype(numpy.float32)
        self.values = numpy.random.uniform(
            0, 1, (n_points, self.channels)
        ).astype(numpy.float32)

        shape = (self.channels,) + self.dimensions
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, values_data, points_data):
        values = chainer.Variable(values_data)
        points = chainer.Variable(points_data)
        y = insert_voxelization_3d(
            values,
            points,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
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
        function = InsertVoxelization3D(
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
        )

        # cpu
        values_cpu = chainer.Variable(self.values)
        points_cpu = chainer.Variable(self.points)
        y_cpu = function(values_cpu, points_cpu)

        # gpu
        values_gpu = chainer.Variable(cuda.to_gpu(self.values))
        points_gpu = chainer.Variable(cuda.to_gpu(self.points))
        y_gpu = function(values_gpu, points_gpu)

        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, values_data, points_data, y_grad):
        gradient_check.check_backward(
            InsertVoxelization3D(
                pitch=self.pitch,
                origin=self.origin,
                dimensions=self.dimensions,
            ),
            (values_data, points_data), y_grad, no_grads=[False, True],
            **self.check_backward_options)

    # @condition.retry(3)
    # def test_backward_cpu(self):
    #     self.check_backward(self.values, self.points, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.values), cuda.to_gpu(self.points),
                            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
'''
