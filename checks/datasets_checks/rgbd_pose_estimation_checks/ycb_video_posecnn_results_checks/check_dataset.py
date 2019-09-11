#!/usr/bin/env python

import objslampp

import imgviz


def get_scene():
    dataset = objslampp.datasets.YCBVideoPoseCNNResultsRGBDPoseEstimationDataset()  # NOQA

    for index in range(len(dataset)):
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
        viz = imgviz.tile(vizs)

        scenes['object_rgb'] = viz

        yield scenes


objslampp.extra.trimesh.display_scenes(
    get_scene(), tile=(1, 2)
)
