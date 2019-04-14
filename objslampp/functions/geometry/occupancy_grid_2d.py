import collections
import numbers

import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np


class OccupancyGrid2D(chainer.Function):

    def __init__(self, *, pitch, origin, dimension):
        assert isinstance(pitch, numbers.Real)
        assert isinstance(origin, collections.Sequence) and len(origin) == 2
        assert isinstance(dimension, collections.Sequence) and \
            len(dimension) == 2

        self.pitch = pitch
        self.origin = origin
        self.dimension = dimension

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        points_type, = in_types
        chainer.utils.type_check.expect(
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 2,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        points, = inputs
        self._points_shape = points.shape
        dtype = points.dtype

        dimension = self.dimension
        origin = xp.asarray(self.origin, dtype=dtype)
        pitch = xp.asarray(self.pitch, dtype=dtype)

        I, J, K = xp.meshgrid(
            xp.arange(dimension[0]),
            xp.arange(dimension[1]),
            xp.arange(points.shape[0]),
        )
        d_IK = I.astype(dtype) - ((points[K, 0] - origin[0]) / pitch)
        d_JK = J.astype(dtype) - ((points[K, 1] - origin[1]) / pitch)
        return d_IK, d_JK

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*grad_outputs)

        grad_d_IK, grad_d_JK = grad_outputs
        dtype = grad_d_IK.dtype

        pitch = xp.asarray(self.pitch, dtype=dtype)
        grad_points_x = (- grad_d_IK / pitch).sum(axis=(0, 1))
        grad_points_y = (- grad_d_JK / pitch).sum(axis=(0, 1))
        grad_points = xp.concatenate(
            (grad_points_x[:, None], grad_points_y[:, None]), axis=1
        )
        return grad_points,


def occupancy_grid_2d(points, *, pitch, origin, dimension, connectivity=1):
    d_IK, d_JK = OccupancyGrid2D(
        pitch=pitch, origin=origin, dimension=dimension
    )(points)
    d_IJK = F.sqrt(d_IK ** 2 + d_JK ** 2)
    m_IJK = F.relu(connectivity - F.absolute(d_IJK))
    m_IJK = F.minimum(m_IJK, m_IJK.array * 0 + 1)
    m = F.max(m_IJK, axis=2)
    return m


if __name__ == '__main__':
    import chainer.gradient_check

    pitch = 1
    origin = (0, 0)
    dimension = (5, 5)
    points = np.array([[0.01, 0.05], [3.99, 3.9]], dtype=np.float32)
    print(f'points:\n{points}')
    points = chainer.Variable(points)
    m_pred = occupancy_grid_2d(
        points,
        pitch=pitch,
        origin=origin,
        dimension=dimension,
    )
    m_true = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)
    print(f'm_pred:\n{m_pred.array}')
    print(f'm_true:\n{m_true}')

    loss = chainer.functions.mean_squared_error(m_pred, m_true)
    loss.backward()

    print(f'points.grad:\n{points.grad}')

    def check_backward(points_data, grad_m):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_2d(
                x,
                pitch=pitch,
                origin=origin,
                dimension=dimension,
            ),
            points_data,
            grad_m,
        )

    grad_m = np.random.uniform(-1, 1, dimension).astype(points.dtype)
    check_backward(points.array, grad_m)
