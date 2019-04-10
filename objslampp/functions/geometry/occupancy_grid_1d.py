import numbers

import chainer
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

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        points, = inputs
        dtype = points.dtype

        dimension = self.dimension
        origin = np.asarray(self.origin, dtype=dtype)
        pitch = np.asarray(self.pitch, dtype=dtype)

        I, J = np.meshgrid(
            np.arange(dimension),
            np.arange(points.shape[0]),
        )
        d_IJ = I.astype(dtype) - ((points[J] - origin) / pitch)
        return d_IJ,

    def backward_cpu(self, inputs, grad_outputs):
        points, = inputs
        dtype = points.dtype

        grad_d_IJ, = grad_outputs
        grad_points = np.zeros_like(points)

        pitch = np.asarray(self.pitch, dtype=dtype)
        for j in range(points.shape[0]):
            grad_points[j] = (- grad_d_IJ[j, :] / pitch).sum()
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
