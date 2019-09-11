#!/usr/bin/env python

import objslampp

import imgviz
import numpy as np
import trimesh


def get_scene(dataset):
    camera = trimesh.scene.Camera(
        fov=(30, 22.5),
        transform=objslampp.extra.trimesh.to_opengl_transform()
    )
    index = 0
    frame = dataset.get_frame(index)
    examples = dataset.get_example(index)

    scenes = {
        'scene_rgb': frame['rgb'],
        'object_rgb': None,
    }

    vizs = []
    for i, example in enumerate(examples):
        viz = imgviz.tile([
            example['rgb'],
            imgviz.depth2rgb(example['pcd'][:, :, 0]),
            imgviz.depth2rgb(example['pcd'][:, :, 1]),
            imgviz.depth2rgb(example['pcd'][:, :, 2]),
        ], border=(255, 255, 255))
        vizs.append(viz)

        geom = trimesh.voxel.Voxel(
            example['grid_target'],
            example['pitch'],
            example['origin'],
        ).as_boxes(colors=(1., 0, 0, 0.5))
        scenes[f'occupied_{i:04d}'] = trimesh.Scene(geom, camera=camera)

        geom = trimesh.voxel.Voxel(
            example['grid_nontarget'],
            example['pitch'],
            example['origin'],
        ).as_boxes(colors=(0, 1., 0, 0.5))
        scenes[f'occupied_{i:04d}'].add_geometry(geom)

        geom = trimesh.voxel.Voxel(
            example['grid_empty'],
            example['pitch'],
            example['origin'],
        ).as_boxes(colors=(0.5, 0.5, 0.5, 0.5))
        scenes[f'empty_{i:04d}'] = trimesh.Scene(geom, camera=camera)

        scenes[f'full_occupied_{i:04d}'] = trimesh.Scene(camera=camera)
        if (example['grid_target_full'] > 0).any():
            geom = trimesh.voxel.Voxel(
                example['grid_target_full'],
                example['pitch'],
                example['origin'],
            ).as_boxes(colors=(1., 0, 0, 0.5))
            scenes[f'full_occupied_{i:04d}'].add_geometry(geom)

        if (example['grid_nontarget_full'] > 0).any():
            colors = imgviz.label2rgb(
                example['grid_nontarget_full'].reshape(1, -1) + 1
            ).reshape(example['grid_nontarget_full'].shape + (3,))
            geom = trimesh.voxel.Voxel(
                example['grid_nontarget_full'],
                example['pitch'],
                example['origin'],
            ).as_boxes(colors=colors)
            scenes[f'full_occupied_{i:04d}'].add_geometry(geom)

        dim = example['grid_target'].shape[0]
        extents = np.array([dim, dim, dim]) * example['pitch']
        geom = trimesh.path.creation.box_outline(extents)
        geom.apply_translation(
            example['origin'] + (dim / 2 - 0.5) * example['pitch']
        )
        scenes[f'occupied_{i:04d}'].add_geometry(geom)
        scenes[f'empty_{i:04d}'].add_geometry(geom)
    viz = imgviz.tile(vizs)

    scenes['object_rgb'] = viz

    return scenes


if __name__ == '__main__':
    dataset = objslampp.datasets.YCBVideoRGBDPoseEstimationDataset(
        split='train'
    )
    objslampp.extra.trimesh.display_scenes(
        get_scene(dataset), height=int(320 * 0.8), width=int(480 * 0.8)
    )
