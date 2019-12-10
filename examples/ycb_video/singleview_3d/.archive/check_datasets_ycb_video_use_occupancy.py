#!/usr/bin/env python

import numpy as np
import trimesh

import morefusion

import contrib


def get_scenes():
    dataset = contrib.datasets.YCBVideoDataset(
        'train',
        class_ids=None,
        augmentation={'rgb', 'depth'},
        return_occupancy_grids=True,
    )
    print(f'dataset_size: {len(dataset)}')

    example = dataset.get_example(0)[0]

    pitch = example['pitch']
    origin = example['origin']

    grid_target = example['grid_target']
    grid_nontarget = example['grid_nontarget']
    grid_empty = example['grid_empty']
    grid_unknown = np.ones_like(example['grid_target'])
    grid_unknown[grid_target > 0] = 0
    grid_unknown[grid_nontarget > 0] = 0
    grid_unknown[grid_empty > 0] = 0

    center = origin + pitch * (np.array(grid_target.shape) / 2 - 0.5)

    camera = trimesh.scene.Camera(
        resolution=(640, 480),
        fov=(60, 45),
        transform=morefusion.extra.trimesh.to_opengl_transform(),
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

    scenes['unknown'] = trimesh.Scene(camera=camera)
    voxel = trimesh.voxel.Voxel(grid_unknown, pitch=pitch, origin=origin)
    geom = voxel.as_boxes(colors=(0, 0, 255))
    scenes['unknown'].add_geometry(geom)
    geom = trimesh.path.creation.box_outline((32 * pitch,) * 3)
    geom.apply_translation(center)
    scenes['unknown'].add_geometry(geom)

    return scenes


def main():
    morefusion.extra.trimesh.display_scenes(
        get_scenes(),
        height=int(round(480 * 0.75)),
        width=int(round(640 * 0.75)),
        tile=(1, 3),
    )


if __name__ == '__main__':
    main()
