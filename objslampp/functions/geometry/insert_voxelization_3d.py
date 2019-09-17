from chainer.backends import cuda
import numpy as np

from .voxelization_3d import Voxelization3D


class InsertVoxelization3D(Voxelization3D):

    def forward_cpu(self, inputs):
        values, points = inputs

        # validation
        if np.isnan(points).sum():
            raise ValueError('points include nan')

        n_points = points.shape[0]
        channels = values.shape[1]

        shape = (channels,) + self.dimensions
        matrix = np.zeros(shape, dtype=np.float32)
        indices = np.full(self.dimensions, -1, dtype=np.int32)

        for i in range(n_points):
            point = points[i]
            value = values[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)

            valid = ((0 <= index) & (index < self.dimensions)).all()
            if not valid:
                continue

            ix, iy, iz = index
            if indices[ix, iy, iz] != -1:
                continue

            matrix[:, ix, iy, iz] = value
            indices[ix, iy, iz] = i

        return matrix,

    def backward_cpu(self, inputs, gy):
        raise NotImplementedError

    def forward_gpu(self, inputs):
        values, points = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        channels = values.shape[1]

        shape = (channels,) + self.dimensions
        matrix = cuda.cupy.zeros(shape, dtype=np.float32)
        indices = cuda.cupy.full(self.dimensions, -1, dtype=np.int32)

        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        dims = cuda.cupy.asarray(self.dimensions, dtype=np.int32)

        n_points = points.shape[0]
        looper = cuda.cupy.empty((n_points,), dtype=np.int8)
        cuda.cupy.ElementwiseKernel(
            '''
            int8 looper, raw float32 values, raw float32 points,
            float32 pitch, raw float32 origin, raw int32 dims, int32 channels
            ''',
            'raw float32 matrix, raw int32 indices',
            r'''
            float x = points[i * 3];
            float y = points[i * 3 + 1];
            float z = points[i * 3 + 2];

            int ix = round((x - origin[0]) / pitch);
            int iy = round((y - origin[1]) / pitch);
            int iz = round((z - origin[2]) / pitch);

            if (ix >= 0 && ix < dims[0] &&
                iy >= 0 && iy < dims[1] &&
                iz >= 0 && iz < dims[2])
            {
                int i_indices = (ix * dims[1] * dims[2]) +
                                (iy * dims[2]) + iz;
                int old_i = atomicCAS(&indices[i_indices], -1, i);
                if (old_i == -1) {
                    for (int c = 0; c < channels; c++) {
                        int i_matrix = (c * dims[0] * dims[1] * dims[2]) +
                                       i_indices;
                        matrix[i_matrix] = values[i * channels + c];
                    }
                }
            }
            ''',
            'voxelize_fwd',
        )(
            looper,
            values,
            points,
            self.pitch,
            origin,
            dims,
            channels,
            matrix,
            indices
        )

        self.n_points = n_points
        self.indices = indices
        return matrix,

    def backward_gpu(self, inputs, gy):
        gmatrix = gy[0]

        channels = gmatrix.shape[0]

        dims = cuda.cupy.asarray(self.dimensions, dtype=np.int32)
        gvalues = cuda.cupy.zeros(
            (self.n_points, channels), dtype=np.float32
        )

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            float32 gmatrix, raw int32 indices, raw int32 dims, int32 channels
            ''',
            'raw float32 gvalues',
            r'''
            // i: index of gmatrix
            int c = i / dims[0] / dims[1] / dims[2];
            int i_indices = i - (c * dims[0] * dims[1] * dims[2]);

            int point_index = indices[i_indices];
            if (point_index >= 0) {
                gvalues[point_index * channels + c] = gmatrix;
            }
            ''',
            'voxelize_bwd',
        )(
            gmatrix, self.indices, dims, channels,
            gvalues,
        )

        return gvalues, None


def insert_voxelization_3d(
    values,
    points,
    *,
    origin,
    pitch,
    dimensions,
    return_indices=False,
):
    return InsertVoxelization3D(
        origin=origin,
        pitch=pitch,
        dimensions=dimensions,
    )(values, points)
