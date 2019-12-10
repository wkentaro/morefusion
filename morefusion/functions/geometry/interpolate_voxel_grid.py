import chainer
from chainer.backends import cuda
import numpy as np


_GET_TRILINEAR_INTERP_KERNEL = '''
__device__
void _get_trilinear_interp_params(
    float ix, float iy, float iz, float weight[8], int ixyz[8][3])
{
    int ix_low = static_cast<int>(ix);
    int iy_low = static_cast<int>(iy);
    int iz_low = static_cast<int>(iz);
    int ix_high = ix_low + 1;
    int iy_high = iy_low + 1;
    int iz_high = iz_low + 1;

    float lx = ix - ix_low;
    float ly = iy - iy_low;
    float lz = iz - iz_low;
    float hx = 1. - lx;
    float hy = 1. - ly;
    float hz = 1. - lz;

    weight[0] = hx * hy * hz;  // w000
    weight[1] = lx * hy * hz;  // w100
    weight[2] = hx * ly * hz;  // w010
    weight[3] = hx * hy * lz;  // w001
    weight[4] = lx * ly * hz;  // w110
    weight[5] = hx * ly * lz;  // w011
    weight[6] = lx * hy * lz;  // w101
    weight[7] = lx * ly * lz;  // w111

    ixyz[0][0] = ix_low;
    ixyz[0][1] = iy_low;
    ixyz[0][2] = iz_low;
    ixyz[1][0] = ix_high;
    ixyz[1][1] = iy_low;
    ixyz[1][2] = iz_low;
    ixyz[2][0] = ix_low;
    ixyz[2][1] = iy_high;
    ixyz[2][2] = iz_low;
    ixyz[3][0] = ix_low;
    ixyz[3][1] = iy_low;
    ixyz[3][2] = iz_high;
    ixyz[4][0] = ix_high;
    ixyz[4][1] = iy_high;
    ixyz[4][2] = iz_low;
    ixyz[5][0] = ix_low;
    ixyz[5][1] = iy_high;
    ixyz[5][2] = iz_high;
    ixyz[6][0] = ix_high;
    ixyz[6][1] = iy_low;
    ixyz[6][2] = iz_high;
    ixyz[7][0] = ix_high;
    ixyz[7][1] = iy_high;
    ixyz[7][2] = iz_high;
}
'''


def _get_trilinear_interp_params(ix, iy, iz):
    ix_low = np.floor(ix).astype(np.int32)
    iy_low = np.floor(iy).astype(np.int32)
    iz_low = np.floor(iz).astype(np.int32)
    ix_high = ix_low + 1
    iy_high = iy_low + 1
    iz_high = iz_low + 1

    lx = ix - ix_low
    ly = iy - iy_low
    lz = iz - iz_low
    hx = 1. - lx
    hy = 1. - ly
    hz = 1. - lz

    weight = np.empty((8,), dtype=np.float32)
    weight[0] = hx * hy * hz  # w000
    weight[1] = lx * hy * hz  # w100
    weight[2] = hx * ly * hz  # w010
    weight[3] = hx * hy * lz  # w001
    weight[4] = lx * ly * hz  # w110
    weight[5] = hx * ly * lz  # w011
    weight[6] = lx * hy * lz  # w101
    weight[7] = lx * ly * lz  # w111

    ixyz = np.empty((8, 3), dtype=np.int32)
    ixyz[0, 0] = ix_low
    ixyz[0, 1] = iy_low
    ixyz[0, 2] = iz_low
    ixyz[1, 0] = ix_high
    ixyz[1, 1] = iy_low
    ixyz[1, 2] = iz_low
    ixyz[2, 0] = ix_low
    ixyz[2, 1] = iy_high
    ixyz[2, 2] = iz_low
    ixyz[3, 0] = ix_low
    ixyz[3, 1] = iy_low
    ixyz[3, 2] = iz_high
    ixyz[4, 0] = ix_high
    ixyz[4, 1] = iy_high
    ixyz[4, 2] = iz_low
    ixyz[5, 0] = ix_low
    ixyz[5, 1] = iy_high
    ixyz[5, 2] = iz_high
    ixyz[6, 0] = ix_high
    ixyz[6, 1] = iy_low
    ixyz[6, 2] = iz_high
    ixyz[7, 0] = ix_high
    ixyz[7, 1] = iy_high
    ixyz[7, 2] = iz_high

    return weight, ixyz


class InterpolateVoxelGrid(chainer.Function):

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 3)

        voxelized_type, points_type, batch_indices_type = in_types
        chainer.utils.type_check.expect(
            voxelized_type.dtype == np.float32,
            voxelized_type.ndim == 5,  # BCXYZ
            points_type.dtype == np.float32,
            points_type.ndim == 2,
            points_type.shape[1] == 3,
            batch_indices_type.dtype == np.int32,
            batch_indices_type.ndim == 1,
            batch_indices_type.shape[0] == points_type.shape[0],
        )

    def forward_cpu(self, x):
        voxelized, points, batch_indices = x

        P, _ = points.shape
        B, C, X, Y, Z = voxelized.shape

        values = np.zeros((P, C), dtype=np.float32)
        for i, point in enumerate(points):
            b = batch_indices[i]
            ix, iy, iz = point
            weight, ixyz = _get_trilinear_interp_params(ix, iy, iz)
            for j in range(8):
                ix, iy, iz = ixyz[j]
                if (ix >= 0 and ix < X and iy >= 0 and iy < Y and
                        iz >= 0 and iz < Z):
                    values[i] += weight[j] * voxelized[b, :, ix, iy, iz]
        return values,

    def backward_cpu(self, x, gy):
        raise NotImplementedError

    def forward_gpu(self, x):
        self.retain_inputs((1, 2))
        voxelized, points, batch_indices = x
        self._shape = voxelized.shape

        P, _ = points.shape
        B, C, X, Y, Z = voxelized.shape

        values = cuda.cupy.zeros((P, C), dtype=np.float32)
        shape = cuda.cupy.array(voxelized.shape, dtype=np.int32)

        cuda.elementwise(
            '''
            raw float32 voxelized, raw float32 points,
            raw int32 batch_indices, raw int32 shape
            ''',
            'float32 values',
            r'''
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            // i: index of values
            // values: (P, C)
            int c = i % C;  // c = {0 ... C}
            int n = i / C;  // n = {0 ... P}
            int b = batch_indices[n];

            float ix = points[n * 3];
            float iy = points[n * 3 + 1];
            float iz = points[n * 3 + 2];

            float weight[8];
            int ixyz[8][3];
            _get_trilinear_interp_params(ix, iy, iz, weight, ixyz);

            for (size_t j = 0; j < 8; j++) {
                if (ixyz[j][0] >= 0 && ixyz[j][0] < X &&
                    ixyz[j][1] >= 0 && ixyz[j][1] < Y &&
                    ixyz[j][2] >= 0 && ixyz[j][2] < Z)
                {
                    int index = (b * C * X * Y * Z) +
                                (c * X * Y * Z) +
                                (ixyz[j][0] * X * Y) +
                                (ixyz[j][1] * Y) +
                                ixyz[j][2];
                    atomicAdd(&values, weight[j] * voxelized[index]);
                }
            }
            ''',
            'interpolate_voxel_grid_fwd',
            preamble=_GET_TRILINEAR_INTERP_KERNEL,
        )(voxelized, points, batch_indices, shape, values)

        return values,

    def backward_gpu(self, x, gy):
        points = x[1]
        batch_indices = x[2]
        gvalues, = gy

        gvoxelized = cuda.cupy.zeros(self._shape, dtype=np.float32)
        shape = cuda.cupy.array(self._shape, dtype=np.int32)

        cuda.elementwise(
            '''
            float32 gvalues, raw float32 points,
            raw int32 batch_indices, raw int32 shape
            ''',
            'raw float32 gvoxelized',
            r'''
            int C = shape[1];
            int X = shape[2];
            int Y = shape[3];
            int Z = shape[4];

            // i: index of gvalues
            // values: (P, C)
            int c = i % C;  // c = {0 ... C}
            int n = i / C;  // n = {0 ... P}
            int b = batch_indices[n];

            float ix = points[n * 3];
            float iy = points[n * 3 + 1];
            float iz = points[n * 3 + 2];

            float weight[8];
            int ixyz[8][3];
            _get_trilinear_interp_params(ix, iy, iz, weight, ixyz);

            for (size_t j = 0; j < 8; j++) {
                if (ixyz[j][0] >= 0 && ixyz[j][0] < X &&
                    ixyz[j][1] >= 0 && ixyz[j][1] < Y &&
                    ixyz[j][2] >= 0 && ixyz[j][2] < Z)
                {
                    int index = (b * C * X * Y * Z) +
                                (c * X * Y * Z) +
                                (ixyz[j][0] * Y * Z) +
                                (ixyz[j][1] * Z) +
                                ixyz[j][2];
                    atomicAdd(&gvoxelized[index], weight[j] * gvalues);
                }
            }
            ''',
            'interpolate_voxel_grid_bwd',
            preamble=_GET_TRILINEAR_INTERP_KERNEL,
        )(gvalues, points, batch_indices, shape, gvoxelized)

        return gvoxelized, None, None


def interpolate_voxel_grid(voxelized, points, batch_indices):
    return InterpolateVoxelGrid()(voxelized, points, batch_indices)


def main():
    import scipy.interpolate

    import morefusion

    dataset = morefusion.datasets.YCBVideoRGBDPoseEstimationDataset('train')
    example = dataset[0][0]

    center = np.nanmean(example['pcd'], axis=(0, 1))
    pitch = 0.0075
    dim = 32
    origin = center - (dim / 2.0 - 0.5) * pitch

    mapping = morefusion.geometry.VoxelMapping(origin, pitch, 32, 3)
    mask = ~np.isnan(example['pcd']).any(axis=2)
    mapping.add(example['pcd'][mask], example['rgb'][mask])

    points = np.array([(15.70, 15.75, 15.75)], dtype=np.float32)

    locs = np.arange(dim), np.arange(dim), np.arange(dim)
    values = scipy.interpolate.RegularGridInterpolator(
        locs, mapping.values
    )(points)
    print(points[0], values[0])

    print('-' * 79)

    voxelized = mapping.values.transpose(3, 0, 1, 2).astype(np.float32)
    values = InterpolateVoxelGrid()(voxelized, points).array
    print(points[0], values[0])

    values = InterpolateVoxelGrid()(
        cuda.to_gpu(voxelized[None]),
        cuda.to_gpu(points),
        cuda.cupy.zeros((points.shape[0],), dtype=np.int32),
    ).array
    print(points[0], values[0])


if __name__ == '__main__':
    main()
