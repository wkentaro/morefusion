import unittest

from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import condition
import trimesh.transformations as tf
import numpy as np

from objslampp.functions.geometry.translation_matrix import translation_matrix


class TestTranslationMatrix(unittest.TestCase):

    def setUp(self):
        batch_size = 5
        self.translations = np.array([
            tf.random_vector((3,)) for _ in range(batch_size)
        ], dtype=np.float32)
        self.transforms = np.array([
            tf.translation_matrix(q) for q in self.translations
        ], dtype=np.float32)
        self.gTs = np.random.uniform(
            -1, 1, (batch_size, 4, 4)
        ).astype(np.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, translations):
        Ts = translation_matrix(translations)
        self.assertEqual(Ts.array.dtype, np.float32)
        self.assertEqual(Ts.shape, self.gTs.shape)

        testing.assert_allclose(
            self.transforms, Ts.array, atol=1e-5, rtol=1e-4
        )

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.translations)

    @testing.attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.translations))

    @testing.attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        y_cpu = translation_matrix(self.translations)
        y_gpu = translation_matrix(cuda.to_gpu(self.translations))
        testing.assert_allclose(y_cpu.array, y_gpu.array)

    def check_backward(self, translations, y_grad):
        gradient_check.check_backward(
            translation_matrix,
            (translations,),
            y_grad,
            **self.check_backward_options,
        )

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.translations, self.gTs)

    @testing.attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.translations), cuda.to_gpu(self.gTs)
        )


testing.run_module(__name__, __file__)
