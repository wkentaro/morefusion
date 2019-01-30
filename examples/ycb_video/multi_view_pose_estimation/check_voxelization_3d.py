#!/usr/bin/env python

from chainer.backends import cuda
import numpy as np
import trimesh
import pybullet  # NOQA

import objslampp


def check_voxelization_3d(gpu):
    np.random.seed(0)

    dataset = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
        split='train'
    )
    data = dataset[0]

    origin = data['scan_origin']
    pitch = data['pitch']
    mask = data['scan_masks'][0]
    pcd = data['scan_pcds'][0]
    rgb = data['scan_rgbs'][0]

    isnan = np.isnan(pcd).any(axis=2)
    points = pcd[(~isnan) & mask].astype(np.float32)
    values = rgb[(~isnan) & mask].astype(np.float32) / 255

    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        values = cuda.to_gpu(values)
        points = cuda.to_gpu(points)

    y = objslampp.functions.voxelization_3d(
        values, points, origin=origin, pitch=pitch, shape=(32, 32, 32, 3)
    )

    matrix_values = cuda.to_cpu(y.array)
    matrix_filled = (matrix_values != 0).any(axis=3)

    geom = trimesh.voxel.Voxel(
        matrix=matrix_filled, pitch=pitch, origin=origin
    ).as_boxes()
    I, J, K = zip(*np.argwhere(matrix_filled))
    geom.visual.face_colors = matrix_values[I, J, K].repeat(12, axis=0)
    geom.show()


def main():
    print('Running on CPU')
    check_voxelization_3d(gpu=-1)
    print('Running on GPU')
    check_voxelization_3d(gpu=0)


if __name__ == '__main__':
    main()
