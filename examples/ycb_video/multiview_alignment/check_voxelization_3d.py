#!/usr/bin/env python

from chainer.backends import cuda
import numpy as np
import pybullet  # NOQA
import trimesh
import trimesh.viewer

import objslampp


def check_voxelization_3d(gpu, **kwargs):
    np.random.seed(0)

    dataset = objslampp.datasets.YCBVideoMultiViewAlignmentDataset(
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
        values,
        points,
        origin=origin,
        pitch=pitch,
        dimensions=(32, 32, 32),
        channels=3,
    )

    matrix_values = cuda.to_cpu(y.array)
    matrix_filled = (matrix_values != 0).any(axis=3)

    scene = trimesh.Scene()
    scene.angles = np.zeros(3)

    geom = trimesh.voxel.Voxel(
        matrix=matrix_filled, pitch=pitch, origin=origin
    ).as_boxes()
    I, J, K = zip(*np.argwhere(matrix_filled))
    geom.visual.face_colors = matrix_values[I, J, K].repeat(12, axis=0)
    scene.add_geometry(geom)

    def callback(scene):
        scene.set_camera(angles=scene.angles)
        scene.angles += [0, np.deg2rad(1), 0]

    trimesh.viewer.SceneViewer(scene=scene, callback=callback, **kwargs)


def main():
    check_voxelization_3d(
        gpu=-1, start_loop=False, caption='Voxelization3D (CPU)'
    )
    check_voxelization_3d(
        gpu=0, caption='Voxelization3D (GPU)'
    )


if __name__ == '__main__':
    main()
