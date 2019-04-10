import collections
import numbers

import chainer
import chainer.functions as F
import numpy as np


class OccupancyGrid3D(chainer.Function):

    def __init__(self, *, pitch, origin, dimension):
        assert isinstance(pitch, numbers.Real)
        assert isinstance(origin, collections.Sequence) and len(origin) == 3
        assert isinstance(dimension, collections.Sequence) and \
            len(dimension) == 3

        self.pitch = pitch
        self.origin = origin
        self.dimension = dimension

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        points_type, = in_types
        chainer.utils.type_check.expect(
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 3,
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((0,))
        points, = inputs
        dtype = points.dtype

        dimension = self.dimension
        origin = np.asarray(self.origin, dtype=dtype)
        pitch = np.asarray(self.pitch, dtype=dtype)

        J, I, K, P = np.meshgrid(
            np.arange(dimension[1]),
            np.arange(dimension[0]),
            np.arange(dimension[2]),
            np.arange(points.shape[0]),
        )
        d_IP = I.astype(dtype) - ((points[P, 0] - origin[0]) / pitch)
        d_JP = J.astype(dtype) - ((points[P, 1] - origin[1]) / pitch)
        d_KP = K.astype(dtype) - ((points[P, 2] - origin[2]) / pitch)
        return d_IP, d_JP, d_KP

    def backward_cpu(self, inputs, grad_outputs):
        points, = inputs
        dtype = points.dtype

        grad_d_IP, grad_d_JP, grad_d_KP = grad_outputs
        grad_points = np.zeros_like(points)

        pitch = np.asarray(self.pitch, dtype=dtype)
        for p in range(points.shape[0]):
            grad_points[p, 0] = (- grad_d_IP[:, :, :, p] / pitch).sum()
            grad_points[p, 1] = (- grad_d_JP[:, :, :, p] / pitch).sum()
            grad_points[p, 2] = (- grad_d_KP[:, :, :, p] / pitch).sum()
        return grad_points,


def occupancy_grid_3d(points, *, pitch, origin, dimension, connectivity=1):
    d_IP, d_JP, d_KP = OccupancyGrid3D(
        pitch=pitch, origin=origin, dimension=dimension
    )(points)
    d_IJKP = F.sqrt(d_IP ** 2 + d_JP ** 2 + d_KP ** 2)
    m_IJKP = F.relu(connectivity - d_IJKP)
    m_IJKP = F.minimum(m_IJKP, m_IJKP.array * 0 + 1)
    m = F.max(m_IJKP, axis=3)
    return m


if __name__ == '__main__':
    import chainer.gradient_check

    pitch = 1
    origin = (0, 0, 0)
    dimension = (5, 5, 5)
    points = np.array(
        [[0, 0.05, 0.1], [3.9, 3.95, 4]],
        dtype=np.float32
    )
    print(f'points:\n{points}')
    points = chainer.Variable(points)
    m_pred = occupancy_grid_3d(
        points,
        pitch=pitch,
        origin=origin,
        dimension=dimension,
    )
    m_true = np.zeros_like(m_pred.array)
    m_true[0, 0, 0] = 1
    m_true[4, 4, 4] = 1
    print(f'm_pred:\n{m_pred.array}')
    print(f'm_true:\n{m_true}')

    loss = chainer.functions.mean_squared_error(m_pred, m_true)
    loss.backward()

    print(f'points.grad:\n{points.grad}')

    def check_backward(points_data, grad_m):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_3d(
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
