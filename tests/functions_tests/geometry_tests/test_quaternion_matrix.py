import unittest

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
import trimesh.transformations as tf
import numpy as np

from objslampp.functions.geometry.quaternion_matrix import quaternion_matrix


class TestQuaternionMatrix(unittest.TestCase):

    def setUp(self):
        self.quaternion = tf.random_quaternion().astype(np.float32)
        self.transform = \
            tf.quaternion_matrix(self.quaternion).astype(np.float32)
        self.gy = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)

    def check_forward(self, quaternion_data):
        quaternion = chainer.Variable(quaternion_data)
        y = quaternion_matrix(quaternion)
        self.assertEqual(y.data.dtype, np.float32)
        self.assertEqual(self.gy.shape, y.shape)

        y_data = cuda.to_cpu(y.data)
        np.testing.assert_allclose(
            self.transform, y_data, atol=1e-5, rtol=1e-4
        )

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.quaternion)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.quaternion))

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        # cpu
        quaternion_cpu = chainer.Variable(self.quaternion)
        y_cpu = quaternion_matrix(quaternion_cpu)

        # gpu
        quaternion_gpu = chainer.Variable(self.quaternion)
        y_gpu = quaternion_matrix(quaternion_gpu)

        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, quaternion_data, y_grad):
        gradient_check.check_backward(
            quaternion_matrix, (quaternion_data,), y_grad
        )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.quaternion, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.quaternion), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
