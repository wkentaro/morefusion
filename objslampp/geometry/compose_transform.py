from chainer.backends import cuda


def compose_transform(R=None, t=None):
    xp = cuda.get_array_module(R, t)

    transform = xp.eye(4)
    if R is not None:
        assert R.shape == (3, 3), 'rotation matrix R must be (3, 3) float'
        transform[:3, :3] = R
    if t is not None:
        assert t.shape == (3,), 'translation vector t must be (3,) float'
        transform[:3, 3] = t

    return transform
