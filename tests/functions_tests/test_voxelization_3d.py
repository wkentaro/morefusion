import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import objslampp


class TestVoxelization3D(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)

        dataset = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
            split='train'
        )
        data = dataset[0]

        self.origin = data['scan_origin']
        self.pitch = data['pitch']

        mask = data['scan_masks'][0]
        pcd = data['scan_pcds'][0]
        rgb = data['scan_rgbs'][0]
        isnan = numpy.isnan(pcd).any(axis=2)
        self.points = pcd[(~isnan) & mask].astype(numpy.float32)
        self.values = rgb[(~isnan) & mask].astype(numpy.float32) / 255

        self.dimensions = (32, 32, 32)
        self.channels = 3
        shape = self.dimensions + (self.channels,)
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, values_data, points_data):
        values = chainer.Variable(values_data)
        points = chainer.Variable(points_data)
        y = objslampp.functions.voxelization_3d(
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
        # cpu
        values_cpu = chainer.Variable(self.values)
        points_cpu = chainer.Variable(self.points)
        y_cpu = objslampp.functions.voxelization_3d(
            values_cpu,
            points_cpu,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
            channels=self.channels,
        )

        # gpu
        values_gpu = chainer.Variable(cuda.to_gpu(self.values))
        points_gpu = chainer.Variable(cuda.to_gpu(self.points))
        y_gpu = objslampp.functions.voxelization_3d(
            values_gpu,
            points_gpu,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
            channels=self.channels,
        )
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, values_data, points_data, y_grad):
        gradient_check.check_backward(
            objslampp.functions.Voxelization3D(
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
