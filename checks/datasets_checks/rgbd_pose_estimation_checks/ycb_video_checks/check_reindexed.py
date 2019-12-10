#!/usr/bin/env python

import os.path as osp

import imgviz

import morefusion


class Images:

    def __init__(self):
        self._dataset_parent = morefusion.datasets.YCBVideoRGBDPoseEstimationDataset('train')  # NOQA
        self._dataset = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed('train')  # NOQA

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        example = self._dataset[index]
        instance_id = self._dataset._ids[index]

        image_id = osp.dirname(instance_id)
        index_parent = self._dataset_parent._ids.index(image_id)
        frame = self._dataset_parent.get_frame(index_parent)

        viz = imgviz.tile([
            example['rgb'],
            imgviz.depth2rgb(example['pcd'][:, :, 0]),
            imgviz.depth2rgb(example['pcd'][:, :, 1]),
            imgviz.depth2rgb(example['pcd'][:, :, 2]),
        ], border=(255, 255, 255))
        viz = imgviz.tile([frame['rgb'], viz], (1, 2))
        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
