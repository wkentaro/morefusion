import unittest

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import condition
import numpy as np
import trimesh.transformations as tf

from morefusion.functions.geometry.quaternion_matrix import quaternion_matrix


class TestQuaternionMatrix(unittest.TestCase):
    def setUp(self):
        batch_size = 5
        self.quaternions = np.array(
            [tf.random_quaternion() for _ in range(batch_size)],
            dtype=np.float32,
        )
        self.transforms = np.array(
            [tf.quaternion_matrix(q) for q in self.quaternions],
            dtype=np.float32,
        )
        self.gTs = np.random.uniform(-1, 1, (batch_size, 4, 4)).astype(
            np.float32
        )
        self.check_backward_options = {"atol": 5e-4, "rtol": 5e-3}

    def check_forward(self, quaternions):
        Ts = quaternion_matrix(quaternions)
        self.assertEqual(Ts.array.dtype, np.float32)
        self.assertEqual(Ts.shape, self.gTs.shape)

        testing.assert_allclose(
            self.transforms, Ts.array, atol=1e-5, rtol=1e-4
        )

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.quaternions)

    @testing.attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.quaternions))

    @testing.attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        y_cpu = quaternion_matrix(self.quaternions)
        y_gpu = quaternion_matrix(cuda.to_gpu(self.quaternions))
        testing.assert_allclose(y_cpu.array, y_gpu.array)

    def check_backward(self, quaternions, y_grad):
        gradient_check.check_backward(
            quaternion_matrix,
            (quaternions,),
            y_grad,
            **self.check_backward_options,
        )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.quaternions, self.gTs)

    @testing.attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.quaternions), cuda.to_gpu(self.gTs)
        )


testing.run_module(__name__, __file__)
