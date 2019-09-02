from chainer.backends import cuda
import numpy as np

from .voxelization_3d import Voxelization3D


class AverageVoxelization3D(Voxelization3D):

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))
        values, points = inputs

        n_points = points.shape[0]

        # validation
        if np.isnan(points).sum():
            raise ValueError('points include nan')

        shape = (self.channels,) + self.dimensions
        matrix = np.zeros(shape, dtype=np.float32)
        counts = np.zeros(self.dimensions, dtype=np.int32)

        for i in range(n_points):
            point = points[i]
            value = values[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)
            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                matrix[:, ix, iy, iz] += value
                counts[ix, iy, iz] += 1

        I, J, K = np.nonzero(counts)
        matrix[:, I, J, K] /= counts[I, J, K]

        self.counts = counts
        return matrix,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        values, points = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        shape = (self.channels,) + self.dimensions
        matrix = cuda.cupy.zeros(shape, dtype=np.float32)
        counts = cuda.cupy.zeros(shape, dtype=np.int32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            float32 values, raw float32 points,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'raw float32 matrix, raw int32 counts',
            r'''
            // i: index of values
            // points: (N, 3)
            // values: (N, C) = (N, shape[0])
            int c = i % shape[0];  // i = {0 ... shape[0]}
            int n = i / shape[0];  // n = {0 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            int ix = static_cast<int>(round((x - origin[0]) / pitch));
            int iy = static_cast<int>(round((y - origin[1]) / pitch));
            int iz = static_cast<int>(round((z - origin[2]) / pitch));

            if (ix >= 0 && ix < shape[1] &&
                iy >= 0 && iy < shape[2] &&
                iz >= 0 && iz < shape[3])
            {
                int index = (c * shape[1] * shape[2] * shape[3]) +
                            (ix * shape[2] * shape[3]) +
                            (iy * shape[3]) + iz;
                atomicAdd(&matrix[index], values);
                atomicAdd(&counts[index], 1);
            }
            ''',
            'voxelize_fwd',
        )(
            values, points,
            self.pitch, origin, shape,
            matrix, counts,
        )

        counts = counts[0]  # counts[i] == count[j]
        I, J, K = cuda.cupy.nonzero(counts)
        matrix[:, I, J, K] /= counts[I, J, K].astype(np.float32)

        self.counts = counts
        return matrix,

    def backward_cpu(self, inputs, gy):
        points = inputs[1]
        counts = self.counts
        gmatrix = gy[0]

        n_points = points.shape[0]

        gvalues = np.zeros((n_points, self.channels), dtype=np.float32)

        for i in range(n_points):
            point = points[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)
            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                gvalues[i] = gmatrix[:, ix, iy, iz] / counts[ix, iy, iz]

        return gvalues, None

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
            raw float32 points, raw float32 gmatrix, raw int32 counts,
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

            int ix = static_cast<int>(round((x - origin[0]) / pitch));
            int iy = static_cast<int>(round((y - origin[1]) / pitch));
            int iz = static_cast<int>(round((z - origin[2]) / pitch));

            if (ix >= 0 && ix < shape[1] &&
                iy >= 0 && iy < shape[2] &&
                iz >= 0 && iz < shape[3])
            {
                int index_counts = (ix * shape[2] * shape[3]) +
                                   (iy * shape[3]) + iz;
                int index_matrix = (c * shape[1] * shape[2] * shape[3]) +
                                   index_counts;
                gvalues = gmatrix[index_matrix] / counts[index_counts];
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
