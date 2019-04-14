import numbers

import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np


class OccupancyGrid1D(chainer.Function):

    def __init__(self, *, pitch, origin, dimension):
        assert isinstance(pitch, numbers.Real)
        assert isinstance(origin, numbers.Real)
        assert isinstance(dimension, int)

        self.pitch = pitch
        self.origin = origin
        self.dimension = dimension

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        points_type, = in_types
        chainer.utils.type_check.expect(
            points_type.dtype == np.float32,
            points_type.ndim == 1,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        points, = inputs
        self._points_shape = points.shape
        dtype = points.dtype

        dimension = self.dimension
        origin = xp.asarray(self.origin, dtype=dtype)
        pitch = xp.asarray(self.pitch, dtype=dtype)

        I, J = xp.meshgrid(
            xp.arange(dimension),
            xp.arange(points.shape[0]),
        )
        d_IJ = I.astype(dtype) - ((points[J] - origin) / pitch)
        return d_IJ,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*grad_outputs)

        grad_d_IJ, = grad_outputs
        dtype = grad_d_IJ.dtype

        pitch = xp.asarray(self.pitch, dtype=dtype)
        grad_points = (- grad_d_IJ / pitch).sum(axis=1)
        return grad_points,


def occupancy_grid_1d(points, *, pitch, origin, dimension):
    assert points.shape == (points.shape[0],)
    d_IJ = OccupancyGrid1D(
        pitch=pitch, origin=origin, dimension=dimension
    )(points)
    m_IJ = F.relu(1 - F.absolute(d_IJ))
    m = F.max(m_IJ, axis=0)
    return m


if __name__ == '__main__':
    import chainer.gradient_check

    pitch = 1
    origin = 0
    dimension = 5
    points = np.array([0.05, 3.9], dtype=np.float32)
    points = chainer.Variable(points)
    m_pred = occupancy_grid_1d(
        points,
        pitch=pitch,
        origin=origin,
        dimension=dimension,
    )
    m_true = np.array([1, 0, 0, 0, 1], dtype=np.float32)
    print(f'm_pred: {m_pred.array.tolist()}')
    print(f'm_true: {m_true.tolist()}')

    loss = chainer.functions.mean_squared_error(m_pred, m_true)
    loss.backward()

    print(f'points.grad: {points.grad}')

    def check_backward(points_data, grad_m):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_1d(
                x,
                pitch=pitch,
                origin=origin,
                dimension=dimension,
            ),
            points_data,
            grad_m,
        )

    grad_m = np.random.uniform(-1, 1, (dimension,)).astype(points.dtype)
    check_backward(points.array, grad_m)
