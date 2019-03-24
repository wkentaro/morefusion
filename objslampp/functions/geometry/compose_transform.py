import chainer
from chainer.backends import cuda


class ComposeTransform(chainer.Function):

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)

        Rs_type, ts_type = in_types
        chainer.utils.type_check.expect(
            Rs_type.ndim == 3,  # N, 3, 3
            Rs_type.shape[0] == ts_type.shape[0],
            Rs_type.shape[1:3] == (3, 3),
            ts_type.ndim == 2,  # N, 3
            ts_type.shape[1] == 3,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        Rs, ts = inputs

        batch_size = Rs.shape[0]
        Ts = xp.eye(4, dtype=Rs.dtype)
        Ts = Ts[None].repeat(batch_size, axis=0)

        Ts[:, :3, :3] = Rs
        Ts[:, :3, 3] = ts
        return Ts,

    def backward(self, inputs, gy):
        gTs, = gy
        gRs = gTs[:, :3, :3]
        gts = gTs[:, :3, 3]
        return gRs, gts


def compose_transform(Rs, ts):
    return ComposeTransform()(Rs, ts)
