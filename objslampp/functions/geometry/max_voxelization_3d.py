from chainer.backends import cuda
import numpy as np

from .voxelization_3d import Voxelization3D


class MaxVoxelization3D(Voxelization3D):

    def forward_cpu(self, inputs):
        values, points, batch_indices, intensities = inputs

        # validation
        if np.isnan(points).sum():
            raise ValueError('points include nan')

        B = self.batch_size
        C = values.shape[1]
        X, Y, Z = self.dimensions

        matrix = np.zeros([B, C, X, Y, Z], dtype=np.float32)
        indices = np.full([B, X, Y, Z], -1, dtype=np.int32)
        max_intensities = np.zeros([B, X, Y, Z], dtype=np.float32)

        for i in range(points.shape[0]):
            batch_index = batch_indices[i]
            point = points[i]
            value = values[i]
            intensity = intensities[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)
            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                if ((indices[batch_index, ix, iy, iz] < 0) or
                        (intensity > max_intensities[batch_index, ix, iy, iz])):  # NOQA
                    matrix[batch_index, :, ix, iy, iz] = value
                    indices[batch_index, ix, iy, iz] = i
                    max_intensities[batch_index, ix, iy, iz] = intensity

        self.indices = indices
        self.n_points = points.shape[0]
        return matrix,

    def backward_cpu(self, inputs, gy):
        gmatrix = gy[0]

        channels = gmatrix.shape[1]
        gvalues = np.zeros((self.n_points, channels), dtype=np.float32)
        for i in range(self.n_points):
            mask = self.indices == i
            ib, ix, iy, iz = np.where(mask)
            gvalues[i] = gmatrix[ib, :, ix, iy, iz].sum(axis=0)

        return gvalues, None, None, None

    def forward_gpu(self, inputs):
        values, points, batch_indices, intensities = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        B = self.batch_size
        C = values.shape[1]
        X, Y, Z = self.dimensions

        matrix = cuda.cupy.zeros([B, C, X, Y, Z], dtype=np.float32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.array([B, C, X, Y, Z], dtype=np.int32)

        indices = cuda.cupy.full([B, X, Y, Z], -1, dtype=np.int32)
        max_intensities = cuda.cupy.zeros([B, X, Y, Z], dtype=np.float32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 values, raw float32 points, int32 batch_indices,
            raw float32 intensities,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            '''
            raw float32 matrix, raw int32 indices,
            raw float32 max_intensities
            ''',
            r'''
            int B = shape[0];
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            int b = batch_indices;
            int n = i;
            float intensity = intensities[n];

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            int ix = static_cast<int>(round((x - origin[0]) / pitch));
            int iy = static_cast<int>(round((y - origin[1]) / pitch));
            int iz = static_cast<int>(round((z - origin[2]) / pitch));

            if (ix >= 0 && ix < X &&
                iy >= 0 && iy < Y &&
                iz >= 0 && iz < Z)
            {
                // BXYZ
                int index = (b * X * Y * Z) + (ix * Y * Z) + (iy * Z) + iz;
                int old_n = atomicCAS(&indices[index], -1, n);
                if (old_n == -1) {
                    atomicExch(&max_intensities[index], intensity);
                } else {
                    float old_intensity = atomicMax(&max_intensities[index],
                                                    intensity);
                    if (intensity > old_intensity) {
                        atomicExch(&indices[index], n);
                    }
                }
            }
            ''',
            'max_voxelization_3d_fwd',
        )(
            values, points, batch_indices, intensities,
            self.pitch, origin, shape,
            matrix, indices, max_intensities,
        )

        valid = indices >= 0
        ib, ix, iy, iz = cuda.cupy.where(valid)
        matrix[ib, :, ix, iy, iz] = values[indices[valid]]

        self.indices = indices
        self.n_points = points.shape[0]

        return matrix,

    def backward_gpu(self, inputs, gy):
        gmatrix = gy[0]
        indices = self.indices

        channels = gmatrix.shape[1]
        shape = cuda.cupy.array(gmatrix.shape, dtype=np.int32)
        gvalues = cuda.cupy.zeros((self.n_points, channels), dtype=np.float32)

        cuda.cupy.ElementwiseKernel(
            '''
            float32 gmatrix, raw int32 indices, raw int32 shape
            ''',
            'raw float32 gvalues',
            r'''
            int B = shape[0];
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            int iz = i % Z;
            int iy = i / Z % Y;
            int ix = i / Z / Y % X;
            int c = i / Z / Y / X % C;
            int b = i / Z / Y / X / C;

            int index1 = (b * X * Y * Z) +
                         (ix * Y * Z) +
                         (iy * Z) +
                         iz;
            int n = indices[index1];
            if (n >= 0) {
                atomicAdd(&gvalues[n * C + c], gmatrix);
            }
            ''',
            'max_voxelization_3d_bwd',
        )(
            gmatrix, indices, shape,
            gvalues,
        )

        return gvalues, None, None, None


def max_voxelization_3d(
    values,
    points,
    batch_indices,
    intensities,
    *,
    batch_size,
    origin,
    pitch,
    dimensions,
    return_indices=False,
):
    func = MaxVoxelization3D(
        batch_size=batch_size,
        origin=origin,
        pitch=pitch,
        dimensions=dimensions,
    )
    voxelized = func(values, points, batch_indices, intensities)
    if return_indices:
        return voxelized, func.indices
    else:
        return voxelized
