from chainer.backends import cuda
import numpy as np
import trimesh

from .voxelization_3d import Voxelization3DBase


class MaxVoxelization3D(Voxelization3DBase):

    def forward_cpu(self, inputs):
        values, points = inputs

        # validation
        if np.isnan(points).sum():
            raise ValueError('points include nan')

        shape = (self.channels,) + self.dimensions
        matrix = np.zeros(shape, dtype=np.float32)
        indices = np.full(shape, -1, dtype=np.int32)  # index of points

        n_points = points.shape[0]
        for i in range(n_points):
            point = points[i]
            value = values[i]

            index = trimesh.voxel.points_to_indices(
                [point], pitch=self.pitch, origin=self.origin
            )[0]

            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                replace = np.where(value > matrix[:, ix, iy, iz])
                indices[:, ix, iy, iz][replace] = i
                matrix[:, ix, iy, iz][replace] = value[replace]

        self.indices = indices
        self.n_points = n_points
        return matrix,

    def backward_cpu(self, inputs, gy):
        gmatrix = gy[0]

        gvalues = np.zeros((self.n_points, self.channels), dtype=np.float32)
        for i in range(self.n_points):
            for j in range(self.channels):
                mask = self.indices[j, :, :, :] == i
                gvalues_i = gmatrix[j, mask]
                assert gvalues_i.size in [0, 1]
                if gvalues_i.size:
                    gvalues[i, j] = gvalues_i

        return gvalues, None

    def forward_gpu(self, inputs):
        values, points = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        shape = (self.channels,) + self.dimensions
        matrix = cuda.cupy.zeros(shape, dtype=np.float32)
        indices = cuda.cupy.full(shape, -1, dtype=np.int32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            float32 values, raw float32 points,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'raw float32 matrix, raw int32 indices',
            r'''
            // i: index of values
            // points: (N, 3)
            // values: (N, C) = (N, shape[0])
            int c = i % shape[0];  // i = {0 ... shape[0]}
            int n = i / shape[0];  // n = {0 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            int ix = round((x - origin[0]) / pitch + 0.5f);
            int iy = round((y - origin[1]) / pitch + 0.5f);
            int iz = round((z - origin[2]) / pitch + 0.5f);

            if (ix >= 0 && ix < shape[1] &&
                iy >= 0 && iy < shape[2] &&
                iz >= 0 && iz < shape[3])
            {
                int index = (c * shape[1] * shape[2] * shape[3]) +
                            (ix * shape[2] * shape[3]) +
                            (iy * shape[3]) + iz;
                float old = atomicMax(&matrix[index], values);
                if (old != values) {
                    atomicExch(&indices[index], i);
                }
            }
            ''',
            'voxelize_fwd',
            preamble='''
            __device__ static float atomicMax(float* address, float val)
            {
                int* address_as_i = (int*) address;
                int old = *address_as_i, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
                } while (assumed != old);
                return __int_as_float(old);
            }
            ''',
        )(
            values, points,
            self.pitch, origin, shape,
            matrix, indices,
        )

        self.indices = indices
        self.n_points = points.shape[0]
        return matrix,

    def backward_gpu(self, inputs, gy):
        indices = self.indices
        gmatrix = gy[0]

        shape = (self.channels,) + self.dimensions
        gvalues = cuda.cupy.zeros(
            (self.n_points, self.channels), dtype=np.float32
        )
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 gmatrix, int32 indices,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'raw float32 gvalues',
            r'''
            if (indices >= 0) {
                // i: index of gmatrix
                // indices: index of gvalues
                // gvalues: (N, C) = (N, shape[0])
                atomicExch(&gvalues[indices], gmatrix[i]);
            }
            ''',
            'voxelize_bwd',
        )(
            gmatrix, indices,
            self.pitch, origin, shape,
            gvalues,
        )

        return gvalues, None


def max_voxelization_3d(
    values,
    points,
    *,
    origin,
    pitch,
    dimensions,
    channels,
):
    return MaxVoxelization3D(
        origin=origin, pitch=pitch, dimensions=dimensions, channels=channels
    )(values, points)
