import chainer
from chainer.backends import cuda
import numpy as np
import trimesh


class Voxelization3D(chainer.Function):

    def __init__(self, *, pitch, origin, shape):
        self.pitch = pitch
        self.origin = origin

        if not (isinstance(shape, tuple) and
                len(shape) == 4 and
                all(isinstance(s, int) for s in shape)):
            raise ValueError('shape must be a tuple of 4 integers')
        self.shape = shape

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)

        values_type, points_type = in_types
        chainer.utils.type_check.expect(
            values_type.dtype == np.float32,
            values_type.ndim == 2,
            values_type.shape[1] == self.shape[3],
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            values_type.shape[0] == points_type.shape[0],
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))
        values, points = inputs

        n_points = points.shape[0]

        # validation
        assert values.shape[0] == n_points, \
            'values and points must be the same length'
        if np.isnan(points).sum():
            raise ValueError('points include nan')

        matrix = np.zeros(self.shape, dtype=np.float32)
        count = np.zeros(self.shape, dtype=np.int32)

        for i in range(n_points):
            point = points[i]
            value = values[i]

            index = trimesh.voxel.points_to_indices(
                [point], pitch=self.pitch, origin=self.origin
            )[0]

            valid = ((0 <= index) & (index < self.shape[:3])).all()
            if valid:
                ix, iy, iz = index
                matrix[ix, iy, iz] += value
                count[ix, iy, iz] += 1

        nonzero = np.nonzero(count)
        matrix[nonzero] /= count[nonzero]

        self.count = count
        return matrix,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        values, points = inputs

        n_points = points.shape[0]

        # validation
        assert values.shape[0] == n_points, \
            'values and points must be the same length'
        if cuda.cupy.isnan(points).sum():
            raise ValueError('points include nan')

        matrix = cuda.cupy.zeros(self.shape, dtype=np.float32)
        count = cuda.cupy.zeros(self.shape, dtype=np.int32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(self.shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            float32 values, raw float32 points,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'raw float32 matrix, raw int32 count',
            r'''
            // i: index of values
            // points: (N, 3)
            // values: (N, C) = (N, shape[3])
            int c = i % shape[3];  // i = {0 ... shape[3]}
            int n = i / shape[3];  // n = {0 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            int ix = round((x - origin[0]) / pitch + 0.5f);
            int iy = round((y - origin[1]) / pitch + 0.5f);
            int iz = round((z - origin[2]) / pitch + 0.5f);

            if (ix >= 0 && ix < shape[0] &&
                iy >= 0 && iy < shape[1] &&
                iz >= 0 && iz < shape[2])
            {
                int index = (ix * shape[1] * shape[2] * shape[3]) +
                            (iy * shape[2] * shape[3]) +
                            (iz * shape[3]) + c;
                atomicAdd(&matrix[index], values);
                atomicAdd(&count[index], 1);
            }
            ''',
            'voxelize_fwd',
        )(
            values, points,
            self.pitch, origin, shape,
            matrix, count,
        )

        nonzero = cuda.cupy.nonzero(count)
        matrix[nonzero] /= count[nonzero].astype(np.float32)

        self.count = count
        return matrix,

    def backward_cpu(self, inputs, gy):
        points = inputs[1]
        count = self.count
        gmatrix = gy[0]

        n_points = points.shape[0]

        gvalues = np.zeros((n_points, self.shape[3]), dtype=np.float32)

        for i in range(n_points):
            point = points[i]

            index = trimesh.voxel.points_to_indices(
                [point], pitch=self.pitch, origin=self.origin
            )[0]

            valid = ((0 <= index) & (index < self.shape[:3])).all()
            if valid:
                ix, iy, iz = index
                gvalues[i] = gmatrix[ix, iy, iz] / count[ix, iy, iz]

        return gvalues, None

    def backward_gpu(self, inputs, gy):
        points = inputs[1]
        count = self.count
        gmatrix = gy[0]

        n_points = points.shape[0]

        gvalues = cuda.cupy.zeros((n_points, self.shape[3]), dtype=np.float32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray(self.shape, dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 points, raw float32 gmatrix, raw int32 count,
            float32 pitch, raw float32 origin, raw int32 shape
            ''',
            'float32 gvalues',
            r'''
            // i: index of gvalues
            // points: (N, 3)
            // gvalues: (N, shape[3])
            int c = i % shape[3];  // i = {0, 1, 2}
            int n = i / shape[3];  // n = {1 ... N}

            float x = points[n * 3];
            float y = points[n * 3 + 1];
            float z = points[n * 3 + 2];

            int ix = round((x - origin[0]) / pitch + 0.5f);
            int iy = round((y - origin[1]) / pitch + 0.5f);
            int iz = round((z - origin[2]) / pitch + 0.5f);

            if (ix >= 0 && ix < shape[0] &&
                iy >= 0 && iy < shape[1] &&
                iz >= 0 && iz < shape[2])
            {
                int index = (ix * shape[1] * shape[2] * shape[3]) +
                            (iy * shape[2] * shape[3]) +
                            (iz * shape[3]) + c;
                gvalues = gmatrix[index] / count[index];
            }
            ''',
            'voxelize_bwd',
        )(
            points, gmatrix, count,
            self.pitch, origin, shape,
            gvalues,
        )

        return gvalues, None


def voxelization_3d(values, points, *, origin, pitch, shape):
    return Voxelization3D(
        origin=origin, pitch=pitch, shape=shape
    )(values, points)
