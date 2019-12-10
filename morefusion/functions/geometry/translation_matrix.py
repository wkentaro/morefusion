import chainer
from chainer.backends import cuda


class TranslationMatrix(chainer.Function):

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        t_type, = in_types
        chainer.utils.type_check.expect(
            t_type.ndim == 2,
            t_type.shape[1] == 3,
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        t, = x

        batch_size = t.shape[0]
        T = xp.eye(4, dtype=t.dtype)[None].repeat(batch_size, axis=0)
        T[:, :3, 3] = t
        return T,

    def backward(self, x, gy):
        gT, = gy
        gt = gT[:, :3, 3]
        return gt,


def translation_matrix(translation):
    squeeze_axis0 = False
    if translation.ndim == 1:
        translation = translation[None]
        squeeze_axis0 = True

    matrix = TranslationMatrix()(translation)

    if squeeze_axis0:
        matrix = matrix[0, :, :]
    return matrix
