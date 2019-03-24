import chainer
import chainer.functions as F
from chainer.backends import cuda


class QuaternionMatrix(chainer.Function):

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        q_type, = in_types
        chainer.utils.type_check.expect(
            q_type.ndim == 3,
            q_type.shape[1:3] == (4, 4),
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        q, = x

        batch_size = q.shape[0]
        R = xp.eye(4, dtype=q.dtype)[None].repeat(batch_size, axis=0)

        R[:, 0, 0] = 1 - q[:, 2, 2] - q[:, 3, 3]
        R[:, 0, 1] = q[:, 1, 2] - q[:, 3, 0]
        R[:, 0, 2] = q[:, 1, 3] + q[:, 2, 0]
        R[:, 1, 0] = q[:, 1, 2] + q[:, 3, 0]
        R[:, 1, 1] = 1 - q[:, 1, 1] - q[:, 3, 3]
        R[:, 1, 2] = q[:, 2, 3] - q[:, 1, 0]
        R[:, 2, 0] = q[:, 1, 3] - q[:, 2, 0]
        R[:, 2, 1] = q[:, 2, 3] + q[:, 1, 0]
        R[:, 2, 2] = 1 - q[:, 1, 1] - q[:, 2, 2]

        return R,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*gy)
        gR, = gy

        batch_size = gR.shape[0]
        gq = xp.zeros((batch_size, 4, 4), dtype=gR.dtype)

        gq[:, 1, 0] = - gR[:, 1, 2] + gR[:, 2, 1]
        gq[:, 1, 1] = - gR[:, 1, 1] - gR[:, 2, 2]
        gq[:, 1, 2] = gR[:, 0, 1] + gR[:, 1, 0]
        gq[:, 1, 3] = gR[:, 0, 2] + gR[:, 2, 0]
        gq[:, 2, 0] = gR[:, 0, 2] - gR[:, 2, 0]
        gq[:, 2, 2] = - gR[:, 0, 0] - gR[:, 2, 2]
        gq[:, 2, 3] = gR[:, 1, 2] + gR[:, 2, 1]
        gq[:, 3, 0] = - gR[:, 0, 1] + gR[:, 1, 0]
        gq[:, 3, 3] = - gR[:, 0, 0] - gR[:, 1, 1]

        return gq,


def outer(a, b):
    assert a.shape[0] == b.shape[0]
    assert a.ndim == 2
    assert b.ndim == 2
    M = a.shape[1]
    N = b.shape[1]
    a = F.repeat(a[:, :, None], N, axis=2)
    b = F.repeat(b[:, None, :], M, axis=1)
    return a * b


def quaternion_matrix(quaternion):
    norm = F.sum(quaternion ** 2, axis=1, keepdims=True)
    quaternion = quaternion * F.sqrt(2. / norm)
    quaternion = outer(quaternion, quaternion)
    return QuaternionMatrix()(quaternion)
