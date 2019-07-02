#!/usr/bin/env python

import numpy as np
import trimesh

import objslampp

import contrib


def get_scenes():
    dataset = contrib.datasets.YCBVideoDataset(
        'train',
        class_ids=None,
        augmentation={'rgb', 'depth'},
        return_occupancy_grids=True,
    )
    print(f'dataset_size: {len(dataset)}')

    example = dataset.get_example(0)

    pitch = example['pitch']
    origin = example['origin']

    grid_target = example['grid_target']
    grid_nontarget = example['grid_nontarget']
    grid_empty = example['grid_empty']

    center = origin + pitch * (np.array(grid_target.shape) / 2 - 0.5)

    camera = trimesh.scene.Camera(
        resolution=(640, 480),
        fov=(60, 45),
        transform=objslampp.extra.trimesh.to_opengl_transform(),
    )

    scenes = {}

    scenes['occupied'] = trimesh.Scene(camera=camera)
    voxel = trimesh.voxel.Voxel(grid_target, pitch=pitch, origin=origin)
    geom = voxel.as_boxes(colors=(255, 0, 0))
    scenes['occupied'].add_geometry(geom)
    voxel = trimesh.voxel.Voxel(grid_nontarget, pitch=pitch, origin=origin)
    geom = voxel.as_boxes(colors=(0, 255, 0))
    scenes['occupied'].add_geometry(geom)
    geom = trimesh.path.creation.box_outline((32 * pitch,) * 3)
    geom.apply_translation(center)
    scenes['occupied'].add_geometry(geom)

    scenes['empty'] = trimesh.Scene(camera=camera)
    voxel = trimesh.voxel.Voxel(grid_empty, pitch=pitch, origin=origin)
    geom = voxel.as_boxes(colors=(127, 127, 127))
    scenes['empty'].add_geometry(geom)
    geom = trimesh.path.creation.box_outline((32 * pitch,) * 3)
    geom.apply_translation(center)
    scenes['empty'].add_geometry(geom)

    return scenes


def main():
    objslampp.extra.trimesh.display_scenes(get_scenes())


if __name__ == '__main__':
    main()
