import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np


class OccupancyGrid3D(chainer.Function):
    def __init__(self, *, pitch, origin, dims):
        pitch = np.asarray(pitch, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        dims = np.asarray(dims, dtype=np.float32)

        assert pitch.ndim == 0
        assert origin.shape == (3,)
        assert dims.shape == (3,)

        self.pitch = pitch
        self.origin = origin
        self.dims = dims

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        (points_type,) = in_types
        chainer.utils.type_check.expect(
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 3,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        (points,) = inputs
        self._points_shape = points.shape
        dtype = points.dtype

        dims = self.dims
        origin = xp.asarray(self.origin, dtype=dtype)
        pitch = xp.asarray(self.pitch, dtype=dtype)

        # a coordinate -> voxel coordinate
        points = (points - origin) / pitch

        J, I, K, P = xp.meshgrid(
            xp.arange(dims[1]),
            xp.arange(dims[0]),
            xp.arange(dims[2]),
            xp.arange(points.shape[0]),
        )
        d_IP = I.astype(dtype) - points[P, 0]
        d_JP = J.astype(dtype) - points[P, 1]
        d_KP = K.astype(dtype) - points[P, 2]
        return d_IP, d_JP, d_KP

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*grad_outputs)

        grad_d_IP, grad_d_JP, grad_d_KP = grad_outputs
        dtype = grad_d_IP.dtype

        pitch = xp.asarray(self.pitch, dtype=dtype)
        grad_points_x = (-grad_d_IP / pitch).sum(axis=(0, 1, 2))
        grad_points_y = (-grad_d_JP / pitch).sum(axis=(0, 1, 2))
        grad_points_z = (-grad_d_KP / pitch).sum(axis=(0, 1, 2))
        grad_points = xp.concatenate(
            (
                grad_points_x[:, None],
                grad_points_y[:, None],
                grad_points_z[:, None],
            ),
            axis=1,
        )
        return (grad_points,)


def occupancy_grid_3d(points, *, pitch, origin, dims, threshold=1):
    d_IP, d_JP, d_KP = OccupancyGrid3D(pitch=pitch, origin=origin, dims=dims)(
        points
    )
    d_IJKP = F.sqrt(d_IP ** 2 + d_JP ** 2 + d_KP ** 2)
    d_IJK = F.min(d_IJKP, axis=3)
    m_IJK = F.relu(threshold - d_IJK)
    m_IJK = F.minimum(m_IJK, m_IJK.array * 0 + 1)
    return m_IJK


if __name__ == "__main__":
    import chainer.gradient_check

    pitch = 1
    origin = (0, 0, 0)
    dims = (5, 5, 5)
    points_array = np.array([[0, 0.05, 0.1], [3.9, 3.95, 4]], dtype=np.float32)
    print(f"points_array:\n{points_array}")
    points = chainer.Variable(points_array)
    m_pred = occupancy_grid_3d(points, pitch=pitch, origin=origin, dims=dims,)
    m_true = np.zeros_like(m_pred.array)
    m_true[0, 0, 0] = 1
    m_true[4, 4, 4] = 1
    print(f"m_pred:\n{m_pred.array}")
    print(f"m_true:\n{m_true}")

    loss = chainer.functions.mean_squared_error(m_pred, m_true)
    loss.backward()

    print(f"points.grad:\n{points.grad}")

    def check_backward(points_data, grad_m):
        chainer.gradient_check.check_backward(
            lambda x: occupancy_grid_3d(
                x, pitch=pitch, origin=origin, dims=dims,
            ),
            points_data,
            grad_m,
        )

    grad_m = np.random.uniform(-1, 1, dims).astype(points.dtype)
    check_backward(points.array, grad_m)
