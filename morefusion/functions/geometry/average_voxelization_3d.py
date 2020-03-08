from chainer.backends import cuda
import numpy as np

from .voxelization_3d import Voxelization3D


class AverageVoxelization3D(Voxelization3D):
    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        values, points, batch_indices = inputs

        # validation
        if np.isnan(points).sum():
            raise ValueError("points include nan")

        B = self.batch_size
        P = points.shape[0]
        C = values.shape[1]
        X, Y, Z = self.dimensions

        matrix = np.zeros([B, C, X, Y, Z], dtype=np.float32)
        counts = np.zeros([B, X, Y, Z], dtype=np.int32)

        for i in range(P):
            batch_index = batch_indices[i]
            point = points[i]
            value = values[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)
            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                matrix[batch_index, :, ix, iy, iz] += value
                counts[batch_index, ix, iy, iz] += 1

        IB, IX, IY, IZ = np.nonzero(counts)
        matrix[IB, :, IX, IY, IZ] /= counts[IB, IX, IY, IZ][:, None]

        self.counts = counts
        return (matrix,)

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        values, points, batch_indices = inputs

        # validation
        if cuda.cupy.isnan(points).sum():
            raise ValueError("points include nan")

        B = self.batch_size
        C = values.shape[1]
        X, Y, Z = self.dimensions

        matrix = cuda.cupy.zeros([B, C, X, Y, Z], dtype=np.float32)
        counts = cuda.cupy.zeros([B, C, X, Y, Z], dtype=np.int32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.array([B, C, X, Y, Z], dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            """
            float32 values, raw float32 points, raw int32 batch_indices,
            float32 pitch, raw float32 origin, raw int32 shape
            """,
            "raw float32 matrix, raw int32 counts",
            r"""
            int B = shape[0];
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            // i: index of values
            // points: (P, 3)
            // values: (P, C)
            int c = i % C;  // c = {0 ... C}
            int n = i / C;  // n = {0 ... P}
            int b = batch_indices[n];

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
                int index = (b * C * X * Y * Z) +
                            (c * X * Y * Z) +
                            (ix * Y * Z) +
                            (iy * Z) +
                            iz;
                atomicAdd(&matrix[index], values);
                atomicAdd(&counts[index], 1);
            }
            """,
            "average_voxelization_3d_fwd",
        )(
            values,
            points,
            batch_indices,
            self.pitch,
            origin,
            shape,
            matrix,
            counts,
        )

        counts = counts[:, 0]  # counts[:, i] == count[j]
        IB, IX, IY, IZ = np.nonzero(counts)
        matrix[IB, :, IX, IY, IZ] /= counts[IB, IX, IY, IZ][:, None]

        self.counts = counts
        return (matrix,)

    def backward_cpu(self, inputs, gy):
        points = inputs[1]
        batch_indices = inputs[2]
        counts = self.counts
        gmatrix = gy[0]

        P = points.shape[0]
        C = gmatrix.shape[1]
        X, Y, Z = self.dimensions

        gvalues = np.zeros((P, C), dtype=np.float32)

        for i in range(P):
            batch_index = batch_indices[i]
            point = points[i]

            index = ((point - self.origin) / self.pitch).round().astype(int)
            valid = ((0 <= index) & (index < self.dimensions)).all()
            if valid:
                ix, iy, iz = index
                gvalues[i] = (
                    gmatrix[batch_index, :, ix, iy, iz]
                    / counts[batch_index, ix, iy, iz]
                )

        return gvalues, None, None

    def backward_gpu(self, inputs, gy):
        points = inputs[1]
        batch_indices = inputs[2]
        counts = self.counts
        gmatrix = gy[0]

        B = self.batch_size
        P = points.shape[0]
        C = gmatrix.shape[1]
        X, Y, Z = self.dimensions

        gvalues = cuda.cupy.zeros((P, C), dtype=np.float32)
        origin = cuda.cupy.asarray(self.origin, dtype=np.float32)
        shape = cuda.cupy.asarray([B, C, X, Y, Z], dtype=np.int32)

        # cuda.elementwise(
        cuda.cupy.ElementwiseKernel(
            """
            raw float32 points, raw int32 batch_indices,
            raw float32 gmatrix, raw int32 counts,
            float32 pitch, raw float32 origin, raw int32 shape
            """,
            "float32 gvalues",
            r"""
            int B = shape[0];
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            // i: index of gvalues
            // points: (P, 3)
            // gvalues: (P, C)
            int c = i % C;  // c = {0 ... C}
            int n = i / C;  // n = {1 ... N}
            int b = batch_indices[n];

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
                int index_counts = (b * X * Y * Z) +
                                   (ix * Y * Z) +
                                   (iy * Z) +
                                   iz;
                int index_matrix = (b * C * X * Y * Z) +
                                   (c * X * Y * Z) +
                                   (ix * Y * Z) +
                                   (iy * Z) +
                                   iz;
                gvalues = gmatrix[index_matrix] / counts[index_counts];
            }
            """,
            "voxelize_bwd",
        )(
            points,
            batch_indices,
            gmatrix,
            counts,
            self.pitch,
            origin,
            shape,
            gvalues,
        )

        return gvalues, None, None


def average_voxelization_3d(
    values,
    points,
    batch_indices,
    *,
    batch_size,
    origin,
    pitch,
    dimensions,
    return_counts=False,
):
    func = AverageVoxelization3D(
        batch_size=batch_size,
        origin=origin,
        pitch=pitch,
        dimensions=dimensions,
    )
    voxel = func(values, points, batch_indices)
    if return_counts:
        return voxel, func.counts
    else:
        return voxel
