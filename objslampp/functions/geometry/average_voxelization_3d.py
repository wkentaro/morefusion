from chainer.backends import cuda
import numpy as np

from .voxelization_3d import Voxelization3D


class AverageVoxelization3D(Voxelization3D):

    # def forward_cpu(self, inputs):
    #     self.retain_inputs((1,))
    #     values, points = inputs
    #
    #     n_points = points.shape[0]
    #
    #     # validation
    #     if np.isnan(points).sum():
    #         raise ValueError('points include nan')
    #
    #     shape = (self.channels,) + self.dimensions
    #     matrix = np.zeros(shape, dtype=np.float32)
    #     counts = np.zeros(self.dimensions, dtype=np.int32)
    #
    #     for i in range(n_points):
    #         point = points[i]
    #         value = values[i]
    #
    #         index = ((point - self.origin) / self.pitch).round().astype(int)
    #         valid = ((0 <= index) & (index < self.dimensions)).all()
    #         if valid:
    #             ix, iy, iz = index
    #             matrix[:, ix, iy, iz] += value
    #             counts[ix, iy, iz] += 1
    #
    #     I, J, K = np.nonzero(counts)
    #     matrix[:, I, J, K] /= counts[I, J, K]
    #
    #     self.counts = counts
    #     return matrix,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        values, points = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        shape = (self.channels,) + self.dimensions
        matrix = cuda.cupy.zeros(shape, dtype=np.float32)
        counts = cuda.cupy.zeros(self.dimensions, dtype=np.float32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            float32 values, raw float32 points,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'raw float32 matrix, raw float32 counts',
            r'''
            // i: index of values
            // points: (N, 3)
            // values: (N, C) = (N, shape[0])
            int c = i % shape[0];  // i = {0 ... shape[0]}
            int n = i / shape[0];  // n = {0 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            float ix = (x - origin[0]) / pitch;
            float iy = (y - origin[1]) / pitch;
            float iz = (z - origin[2]) / pitch;

            float ix_low = static_cast<int>(ix);
            float iy_low = static_cast<int>(iy);
            float iz_low = static_cast<int>(iz);

            float lx = ix - ix_low;
            float ly = iy - iy_low;
            float lz = iz - iz_low;
            float hx = 1. - lx;
            float hy = 1. - ly;
            float hz = 1. - lz;

            float w[8];
            w[0] = hx * hy * hz;  // w000
            w[1] = lx * hy * hz;  // w100
            w[2] = hx * ly * hz;  // w010
            w[3] = hx * hy * lz;  // w001
            w[4] = lx * ly * hz;  // w110
            w[5] = hx * ly * lz;  // w011
            w[6] = lx * hy * lz;  // w101
            w[7] = lx * ly * lz;  // w111

            int ixyz[8][3];
            ixyz[0][0] = ix_low;
            ixyz[0][1] = iy_low;
            ixyz[0][2] = iz_low;
            ixyz[1][0] = ix_low + 1;
            ixyz[1][1] = iy_low;
            ixyz[1][2] = iz_low;
            ixyz[2][0] = ix_low;
            ixyz[2][1] = iy_low + 1;
            ixyz[2][2] = iz_low;
            ixyz[3][0] = ix_low;
            ixyz[3][1] = iy_low;
            ixyz[3][2] = iz_low + 1;
            ixyz[4][0] = ix_low + 1;
            ixyz[4][1] = iy_low + 1;
            ixyz[4][2] = iz_low;
            ixyz[5][0] = ix_low;
            ixyz[5][1] = iy_low + 1;
            ixyz[5][2] = iz_low + 1;
            ixyz[6][0] = ix_low + 1;
            ixyz[6][1] = iy_low;
            ixyz[6][2] = iz_low + 1;
            ixyz[7][0] = ix_low + 1;
            ixyz[7][1] = iy_low + 1;
            ixyz[7][2] = iz_low + 1;

            for (size_t j = 0; j < 8; j++) {
                if (ixyz[j][0] >= 0 && ixyz[j][0] < shape[1] &&
                    ixyz[j][1] >= 0 && ixyz[j][1] < shape[2] &&
                    ixyz[j][2] >= 0 && ixyz[j][2] < shape[3])
                {
                    int index_counts = (ixyz[j][0] * shape[2] * shape[3]) +
                                       (ixyz[j][1] * shape[3]) + ixyz[j][2];
                    int index_matrix = (c * shape[1] * shape[2] * shape[3]) +
                                       index_counts;
                    atomicAdd(&matrix[index_matrix], w[j] * values);
                    atomicAdd(&counts[index_counts], w[j]);
                }
            }
            ''',
            'voxelize_fwd',
        )(
            values, points,
            self.pitch, origin, shape,
            matrix, counts,
        )

        I, J, K = cuda.cupy.nonzero(counts)
        matrix[:, I, J, K] /= counts[I, J, K].astype(np.float32)

        self.counts = counts
        return matrix,

    # def backward_cpu(self, inputs, gy):
    #     points = inputs[1]
    #     counts = self.counts
    #     gmatrix = gy[0]
    #
    #     n_points = points.shape[0]
    #
    #     gvalues = np.zeros((n_points, self.channels), dtype=np.float32)
    #
    #     for i in range(n_points):
    #         point = points[i]
    #
    #         index = ((point - self.origin) / self.pitch).round().astype(int)
    #         valid = ((0 <= index) & (index < self.dimensions)).all()
    #         if valid:
    #             ix, iy, iz = index
    #             gvalues[i] = gmatrix[:, ix, iy, iz] / counts[ix, iy, iz]
    #
    #     return gvalues, None

    def backward_gpu(self, inputs, gy):
        points = inputs[1]
        counts = self.counts
        gmatrix = gy[0]

        n_points = points.shape[0]

        shape = (self.channels,) + self.dimensions
        gvalues = cuda.cupy.zeros((n_points, self.channels), dtype=np.float32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 points, raw float32 gmatrix, raw float32 counts,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'float32 gvalues',
            r'''
            // i: index of gvalues
            // points: (N, 3)
            // gvalues: (N, C) = (N, shape[0])
            int c = i % shape[0];  // i = {0 ... shape[0]}
            int n = i / shape[0];  // n = {1 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            float ix = (x - origin[0]) / pitch;
            float iy = (y - origin[1]) / pitch;
            float iz = (z - origin[2]) / pitch;

            float ix_low = static_cast<int>(ix);
            float iy_low = static_cast<int>(iy);
            float iz_low = static_cast<int>(iz);

            float lx = ix - ix_low;
            float ly = iy - iy_low;
            float lz = iz - iz_low;
            float hx = 1. - lx;
            float hy = 1. - ly;
            float hz = 1. - lz;

            float w[8];
            w[0] = hx * hy * hz;  // w000
            w[1] = lx * hy * hz;  // w100
            w[2] = hx * ly * hz;  // w010
            w[3] = hx * hy * lz;  // w001
            w[4] = lx * ly * hz;  // w110
            w[5] = hx * ly * lz;  // w011
            w[6] = lx * hy * lz;  // w101
            w[7] = lx * ly * lz;  // w111

            int ixyz[8][3];
            ixyz[0][0] = ix_low;
            ixyz[0][1] = iy_low;
            ixyz[0][2] = iz_low;
            ixyz[1][0] = ix_low + 1;
            ixyz[1][1] = iy_low;
            ixyz[1][2] = iz_low;
            ixyz[2][0] = ix_low;
            ixyz[2][1] = iy_low + 1;
            ixyz[2][2] = iz_low;
            ixyz[3][0] = ix_low;
            ixyz[3][1] = iy_low;
            ixyz[3][2] = iz_low + 1;
            ixyz[4][0] = ix_low + 1;
            ixyz[4][1] = iy_low + 1;
            ixyz[4][2] = iz_low;
            ixyz[5][0] = ix_low;
            ixyz[5][1] = iy_low + 1;
            ixyz[5][2] = iz_low + 1;
            ixyz[6][0] = ix_low + 1;
            ixyz[6][1] = iy_low;
            ixyz[6][2] = iz_low + 1;
            ixyz[7][0] = ix_low + 1;
            ixyz[7][1] = iy_low + 1;
            ixyz[7][2] = iz_low + 1;

            for (size_t j = 0; j < 8; j++) {
                if (ixyz[j][0] >= 0 && ixyz[j][0] < shape[1] &&
                    ixyz[j][1] >= 0 && ixyz[j][1] < shape[2] &&
                    ixyz[j][2] >= 0 && ixyz[j][2] < shape[3])
                {
                    int index_counts = (ixyz[j][0] * shape[2] * shape[3]) +
                                       (ixyz[j][1] * shape[3]) + ixyz[j][2];
                    int index_matrix = (c * shape[1] * shape[2] * shape[3]) +
                                       index_counts;
                    gvalues += w[j] * gmatrix[index_matrix] /
                               counts[index_counts];
                }
            }
            ''',
            'voxelize_bwd',
        )(
            points, gmatrix, counts,
            self.pitch, origin, shape,
            gvalues,
        )

        return gvalues, None


def average_voxelization_3d(
    values,
    points,
    *,
    origin,
    pitch,
    dimensions,
    return_counts=False,
):
    channels = values.shape[1]
    func = AverageVoxelization3D(
        origin=origin, pitch=pitch, dimensions=dimensions, channels=channels
    )
    voxel = func(values, points)
    if return_counts:
        return voxel, func.counts
    else:
        return voxel
