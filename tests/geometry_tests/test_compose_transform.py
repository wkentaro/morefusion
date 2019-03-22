import numpy as np
import trimesh.transformations as tf

from objslampp.geometry.compose_transform import compose_transform


def test_compose_transform():
    R = tf.random_rotation_matrix()
    t = tf.random_vector((3,))

    T = compose_transform(R=R[:3, :3])
    np.testing.assert_allclose(T, R)

    T = compose_transform(t=t)
    np.testing.assert_allclose(T, tf.translation_matrix(t))

    T = compose_transform(R=R[:3, :3], t=t)
    np.testing.assert_allclose(T, tf.translation_matrix(t) @ R)
