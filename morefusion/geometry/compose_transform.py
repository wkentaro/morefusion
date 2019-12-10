import chainer
from chainer.backends import cuda

from ..functions import compose_transform as compose_transform_function


def compose_transform(R=None, t=None):
    xp = cuda.get_array_module(R, t)

    if R is None:
        Rs = xp.eye(3)[None]
    else:
        Rs = R[None]

    if t is None:
        ts = xp.zeros((1, 3))
    else:
        ts = t[None]

    with chainer.no_backprop_mode():
        Ts = compose_transform_function(Rs, ts).array
        T = Ts[0]

    return T
