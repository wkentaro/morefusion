#!/usr/bin/env python

import imgviz

import objslampp


class Images:

    def __init__(self):
        self._dataset = objslampp.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed('train')  # NOQA

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        example = self._dataset[index]
        instance_id = self._dataset._ids[index]
        print(instance_id)
        viz = imgviz.tile([
            example['rgb'],
            imgviz.depth2rgb(example['pcd'][:, :, 0]),
            imgviz.depth2rgb(example['pcd'][:, :, 1]),
            imgviz.depth2rgb(example['pcd'][:, :, 2]),
        ], border=(255, 255, 255))
        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
