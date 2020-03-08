import unittest

from chainer.backends import cuda
from chainer import testing
import trimesh.transformations as tf

from morefusion.geometry.compose_transform import compose_transform


class TestComposeTransform(unittest.TestCase):
    def setUp(self):
        self.R = tf.random_rotation_matrix()
        self.t = tf.random_vector((3,))

    def check_compose_transform(self, R, t):
        T = compose_transform(R=R[:3, :3])
        testing.assert_allclose(T, R)

        T = compose_transform(t=t)
        testing.assert_allclose(T, tf.translation_matrix(cuda.to_cpu(t)))

        T = compose_transform(R=R[:3, :3], t=t)
        testing.assert_allclose(
            T, tf.translation_matrix(cuda.to_cpu(t)) @ cuda.to_cpu(R)
        )

    def test_compose_transform_cpu(self):
        self.check_compose_transform(self.R, self.t)

    @testing.attr.gpu
    def test_compose_transform_gpu(self):
        self.check_compose_transform(cuda.to_gpu(self.R), cuda.to_gpu(self.t))
