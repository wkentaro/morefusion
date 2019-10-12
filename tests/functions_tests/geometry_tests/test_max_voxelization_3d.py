import unittest

import numpy

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from objslampp.functions.geometry.max_voxelization_3d \
    import max_voxelization_3d
from objslampp.functions.geometry.max_voxelization_3d \
    import MaxVoxelization3D


class TestMaxVoxelization3D(unittest.TestCase):

    def setUp(self):
        voxel_dim = 32
        self.dimensions = (voxel_dim, voxel_dim, voxel_dim)
        self.channels = 4

        self.origin = numpy.array([-1, -1, -1], dtype=numpy.float32)
        self.pitch = numpy.float32(2. / voxel_dim)

        self.batch_size = 3
        n_points = 128
        self.points = numpy.random.uniform(
            -1, 1, (n_points, 3)
        ).astype(numpy.float32)
        self.values = numpy.random.uniform(
            -1, 1, (n_points, self.channels)
        ).astype(numpy.float32)
        self.intensities = numpy.linalg.norm(self.values, axis=1)
        self.batch_indices = numpy.random.randint(
            0, self.batch_size, n_points
        ).astype(numpy.int32)

        shape = (self.batch_size, self.channels) + self.dimensions
        self.gy = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(
        self, values_data, points_data, batch_indices_data, intensities_data
    ):
        y = max_voxelization_3d(
            values_data,
            points_data,
            batch_indices_data,
            intensities_data,
            batch_size=self.batch_size,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
        )
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(
            self.values, self.points, self.batch_indices, self.intensities
        )

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.values),
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.batch_indices),
            cuda.to_gpu(self.intensities),
        )

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        # cpu
        y_cpu = max_voxelization_3d(
            self.values,
            self.points,
            self.batch_indices,
            self.intensities,
            batch_size=self.batch_size,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
        )

        # gpu
        y_gpu = max_voxelization_3d(
            cuda.to_gpu(self.values),
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.batch_indices),
            cuda.to_gpu(self.intensities),
            batch_size=self.batch_size,
            origin=self.origin,
            pitch=self.pitch,
            dimensions=self.dimensions,
        )

        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(
        self,
        values_data,
        points_data,
        batch_indices_data,
        intensities_data,
        y_grad
    ):
        gradient_check.check_backward(
            MaxVoxelization3D(
                batch_size=self.batch_size,
                pitch=self.pitch,
                origin=self.origin,
                dimensions=self.dimensions,
            ),
            (values_data, points_data, batch_indices_data, intensities_data),
            y_grad,
            no_grads=[False, True, True, True],
            **self.check_backward_options,
        )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(
            self.values,
            self.points,
            self.batch_indices,
            self.intensities,
            self.gy,
        )

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.values),
            cuda.to_gpu(self.points),
            cuda.to_gpu(self.batch_indices),
            cuda.to_gpu(self.intensities),
            cuda.to_gpu(self.gy),
        )


testing.run_module(__name__, __file__)
