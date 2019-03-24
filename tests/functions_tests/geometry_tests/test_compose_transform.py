import unittest

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import condition
import numpy as np
import trimesh.transformations as tf

from objslampp.functions.geometry.compose_transform import compose_transform


class TestComposeTransform(unittest.TestCase):

    def setUp(self):
        batch_size = 5
        self.Rs = np.asarray([
            tf.random_rotation_matrix() for _ in range(batch_size)
        ], dtype=np.float32)[:, :3, :3]
        self.ts = np.asarray([
            tf.random_vector((3,)) for _ in range(batch_size)
        ], dtype=np.float32)
        self.gTs = np.random.uniform(
            -1, 1, (batch_size, 4, 4)
        ).astype(np.float32)

    def check_forward(self, Rs, ts):
        Ts = compose_transform(Rs, ts)
        assert Ts.array.dtype == np.float32
        assert Ts.shape == self.gTs.shape

        testing.assert_allclose(Ts.array[:, :3, :3], Rs)
        testing.assert_allclose(Ts.array[:, :3, 3], ts)

    def test_forward_cpu(self):
        self.check_forward(self.Rs, self.ts)

    @testing.attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.Rs), cuda.to_gpu(self.ts))

    @testing.attr.gpu
    def test_forward_cpu_gpu_equal(self):
        y_cpu = compose_transform(self.Rs, self.ts)
        y_gpu = compose_transform(cuda.to_gpu(self.Rs), cuda.to_gpu(self.ts))
        testing.assert_allclose(y_cpu.array, cuda.to_cpu(y_gpu.array))

    def check_backward(self, Rs, ts, y_grad):
        gradient_check.check_backward(compose_transform, (Rs, ts), y_grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.Rs, self.ts, self.gTs)

    @testing.attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.Rs), cuda.to_gpu(self.ts), cuda.to_gpu(self.gTs)
        )
