import chainer
import chainer.functions as F
from chainer.backends import cuda


class QuaternionMatrix(chainer.Function):

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        q, = x
        assert q.shape == (4, 4)

        matrix = xp.array([
            [1 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0],
            [q[1, 2] + q[3, 0], 1 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1 - q[1, 1] - q[2, 2], 0],
            [0, 0, 0, 1]
        ], dtype=q.dtype)
        return matrix,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*gy)
        gmat, = gy
        assert gmat.shape == (4, 4)

        gq = xp.zeros((4, 4), dtype=gmat.dtype)
        gq[1, 0] = - gmat[1, 2] + gmat[2, 1]
        gq[1, 1] = - gmat[1, 1] - gmat[2, 2]
        gq[1, 2] = gmat[0, 1] + gmat[1, 0]
        gq[1, 3] = gmat[0, 2] + gmat[2, 0]
        gq[2, 0] = gmat[0, 2] - gmat[2, 0]
        gq[2, 2] = - gmat[0, 0] - gmat[2, 2]
        gq[2, 3] = gmat[1, 2] + gmat[2, 1]
        gq[3, 0] = - gmat[0, 1] + gmat[1, 0]
        gq[3, 3] = - gmat[0, 0] - gmat[1, 1]
        return gq,


def outer(a, b):
    assert a.ndim == 1
    assert b.ndim == 1
    M = a.shape[0]
    N = b.shape[0]
    a = F.repeat(a[:, None], N, axis=1)
    b = F.repeat(b[None, :], M, axis=0)
    return a * b


def quaternion_matrix(quaternion):
    norm = F.matmul(quaternion, quaternion)
    quaternion = quaternion * F.sqrt(2. / norm)
    quaternion = outer(quaternion, quaternion)
    return QuaternionMatrix()(quaternion)
