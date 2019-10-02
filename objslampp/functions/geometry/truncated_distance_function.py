import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np


class TruncatedDistanceFunction(chainer.Function):

    def __init__(self, *, pitch, origin, dims, truncation):
        self.pitch = pitch
        self.origin = origin
        self.dims = dims
        self.truncation = truncation

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 1)

        points_type, = in_types
        chainer.utils.type_check.expect(
            points_type.ndim == 2,
            points_type.shape[1] == 3,
        )

    def forward_gpu(self, inputs):
        cupy = cuda.cupy

        self.retain_inputs((0,))
        points, = inputs
        dtype = points.dtype

        pitch = cupy.asarray(self.pitch, dtype=dtype)
        origin = cupy.asarray(self.origin, dtype=dtype)
        dims = cupy.asarray(self.dims, dtype=np.int32)
        truncation = cupy.asarray(self.truncation, dtype=dtype)

        matrix = cupy.full(self.dims, self.truncation, dtype=dtype)
        indices = cupy.full(self.dims, -1, dtype=np.int32)

        ksize = int(cupy.ceil(truncation / pitch))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cupy.meshgrid(*(cupy.arange(ksize),) * 3)
        kernel = cupy.stack(kernel, -1).reshape(-1, 3).astype(np.float32)
        kernel -= ksize // 2

        indexer = cupy.empty((points.shape[0], ksize ** 3), dtype=np.int8)
        cuda.elementwise(
            '''
            int8 indexer, raw T points, T pitch,
            raw T origin, raw int32 dims, T truncation,
            int32 ksize, raw T kernel
            ''',
            'raw T matrix, raw int32 indices',
            '''
            int K = ksize * ksize * ksize;
            int k = i % K;
            int p = i / K;

            T ix_f = (points[3 * p] - origin[0]) / pitch;
            T iy_f = (points[3 * p + 1] - origin[1]) / pitch;
            T iz_f = (points[3 * p + 2] - origin[2]) / pitch;

            int ix = round(ix_f) + kernel[3 * k];
            int iy = round(iy_f) + kernel[3 * k + 1];
            int iz = round(iz_f) + kernel[3 * k + 2];

            if (ix >= 0 && ix < dims[0] &&
                    iy >= 0 && iy < dims[1] &&
                    iz >= 0 && iz < dims[2])
            {
                T idx = ix_f - ix;
                T idy = iy_f - iy;
                T idz = iz_f - iz;
                T distance = pitch * sqrt(idx * idx + idy * idy + idz * idz);
                if (distance < truncation) {
                    int index = (ix * dims[1] * dims[2]) + (iy * dims[2]) + iz;
                    T old = atomicMin(&matrix[index], distance);
                    if (old != distance) {
                        atomicExch(&indices[index], i);
                    }
                }
            }
            ''',
            'truncated_distance_function_fwd',
            # preamble='''
            # __device__ static float atomicMin(float* address, float val)
            # {
            #     int* address_as_i = (int*) address;
            #     int old = *address_as_i, assumed;
            #     do {
            #         assumed = old;
            #         old = ::atomicCAS(address_as_i, assumed,
            #             __float_as_int(::fminf(val, __int_as_float(assumed))));
            #     } while (assumed != old);
            #     return __int_as_float(old);
            # }
            # ''',
        )(
            indexer,
            points,
            pitch,
            origin,
            dims,
            truncation,
            ksize,
            kernel,
            matrix,
            indices,
        )

        self._pitch = pitch
        self._origin = origin
        self._indices = indices
        self._ksize = ksize
        self._kernel = kernel

        return matrix,

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy

        points, = inputs
        gmatrix, = grad_outputs
        dtype = points.dtype

        gpoints = cupy.zeros(points.shape, dtype=dtype)

        cuda.elementwise(
            '''
            T gmatrix, raw T points, int32 indices,
            T pitch, raw T origin,
            int32 ksize, raw T kernel
            ''',
            'raw T gpoints',
            '''
            if (indices < 0) {
                return;
            }

            int K = ksize * ksize * ksize;
            int k = indices % K;
            int p = indices / K;

            T ix_f = (points[3 * p] - origin[0]) / pitch;
            T iy_f = (points[3 * p + 1] - origin[1]) / pitch;
            T iz_f = (points[3 * p + 2] - origin[2]) / pitch;

            int ix = round(ix_f) + kernel[3 * k];
            int iy = round(iy_f) + kernel[3 * k + 1];
            int iz = round(iz_f) + kernel[3 * k + 2];

            T idx = ix_f - ix;
            T idy = iy_f - iy;
            T idz = iz_f - iz;

            T idistance = sqrt(idx * idx + idy * idy + idz * idz);
            if (idistance > 0) {
                atomicAdd(&gpoints[3 * p], idx / idistance * gmatrix);
                atomicAdd(&gpoints[3 * p + 1], idy / idistance * gmatrix);
                atomicAdd(&gpoints[3 * p + 2], idz / idistance * gmatrix);
            }
            ''',
            'truncated_distance_function_bwd',
        )(
            gmatrix,
            points,
            self._indices,
            self._pitch,
            self._origin,
            self._ksize,
            self._kernel,
            gpoints,
        )

        K = self._ksize ** 3
        keep = self._indices >= 0
        kernel_indices = self._indices.copy()
        point_indices = self._indices.copy()
        kernel_indices[keep] = self._indices[keep] % K
        point_indices[keep] = self._indices[keep] / K

        return gpoints,


def truncated_distance_function(points, *, pitch, origin, dims, truncation):
    return TruncatedDistanceFunction(
        pitch=pitch,
        origin=origin,
        dims=dims,
        truncation=truncation,
    )(points)


def pseudo_occupancy_voxelization(
    points, *, pitch, origin, dims, threshold=1,
):
    truncation = threshold * pitch
    matrix = truncated_distance_function(
        points,
        pitch=pitch,
        origin=origin,
        dims=dims,
        truncation=truncation,
    )  # [0, truncation]
    matrix = (truncation - matrix) / pitch
    return F.minimum(matrix, matrix * 0 + 1)


if __name__ == '__main__':
    gpu = 0
    xp = cuda.cupy

    import chainer.gradient_check

    pitch = 0.5
    origin = (0, 0, 0)
    dims = (5, 5, 5)
    points = np.array([
        [0.5, 0.5, 0.5],
        [1.48, 1.48, 1.48],
    ], dtype=np.float32)
    points = cuda.to_gpu(points)
    print(f'points:\n{points}')
    points = chainer.Variable(points)
    truncation = 1.2
    m_pred = truncated_distance_function(
        points,
        pitch=pitch,
        origin=origin,
        dims=dims,
        truncation=truncation,
    )
    m_true = m_pred.array.copy()
    m_true += xp.random.uniform(-0.2, 0.2, m_true.shape)
    print(f'm_pred:\n{m_pred.array}')
    print(f'm_true:\n{m_true}')

    loss = chainer.functions.mean_squared_error(m_pred, m_true)
    loss.backward()

    print(f'points.grad:\n{points.grad}')

    def check_backward(points_data, grad_matrix):
        chainer.gradient_check.check_backward(
            func=lambda x: truncated_distance_function(
                x,
                pitch=pitch,
                origin=origin,
                dims=dims,
                truncation=truncation,
            ),
            x_data=points_data,
            y_grad=grad_matrix,
        )

    grad_matrix = xp.random.uniform(-1, 1, dims).astype(points.dtype)
    check_backward(points.array, grad_matrix)
