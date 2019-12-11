#!/usr/bin/env python

from chainer.backends import cuda
import numpy as np
import pyglet
import trimesh
import trimesh.transformations as ttf
import trimesh.viewer

import morefusion


def check_average_voxelization_3d(
    origin, pitch, points, values, gpu, **kwargs
):
    batch_indices = np.zeros((points.shape[0],), dtype=np.int32)

    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        values = cuda.to_gpu(values)
        points = cuda.to_gpu(points)
        batch_indices = cuda.to_gpu(batch_indices)

    y = morefusion.functions.average_voxelization_3d(
        values,
        points,
        batch_indices,
        batch_size=1,
        origin=origin,
        pitch=pitch,
        dimensions=(32, 32, 32),
    )[0]
    y = y.transpose(1, 2, 3, 0)

    matrix_values = cuda.to_cpu(y.array)
    matrix_filled = (matrix_values != 0).any(axis=3)

    scene = trimesh.Scene()
    scene.angles = np.zeros(3)

    geom = trimesh.voxel.VoxelGrid(
        matrix_filled, ttf.scale_and_translate(pitch, origin)
    ).as_boxes()
    I, J, K = zip(*np.argwhere(matrix_filled))
    geom.visual.face_colors = matrix_values[I, J, K].repeat(12, axis=0)
    scene.add_geometry(geom)

    def callback(scene):
        scene.set_camera(angles=scene.angles)
        scene.angles += [0, np.deg2rad(1), 0]

    trimesh.viewer.SceneViewer(scene=scene, callback=callback, **kwargs)


def main():
    dataset = morefusion.datasets.YCBVideoDataset(split='train')
    class_names = morefusion.datasets.ycb_video.class_names

    example = dataset[1000]

    rgb = example['color']
    depth = example['depth']

    K = example['meta']['intrinsic_matrix']
    pcd = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )

    class_id = example['meta']['cls_indexes'][0]
    mask = example['label'] == class_id
    mask = (~np.isnan(pcd).any(axis=2)) & mask

    points = pcd[mask].astype(np.float32)
    values = rgb[mask].astype(np.float32) / 255

    # pitch
    models = morefusion.datasets.YCBVideoModels()
    pitch = models.get_voxel_pitch(dimension=32, class_id=class_id)

    # origin
    centroid = points.mean(axis=0)
    origin = centroid - pitch * 32 / 2.

    print(f'class_id: {class_id}')
    print(f'class_name: {class_names[class_id]}')
    print(f'origin: {origin}')
    print(f'pitch: {pitch}')

    check_average_voxelization_3d(
        origin,
        pitch,
        points,
        values,
        gpu=-1,
        start_loop=False,
        caption='Voxelization3D (CPU)',
        resolution=(400, 400),
    )
    check_average_voxelization_3d(
        origin,
        pitch,
        points,
        values,
        gpu=0,
        start_loop=False,
        caption='Voxelization3D (GPU)',
        resolution=(400, 400),
    )
    pyglet.app.run()


if __name__ == '__main__':
    main()
