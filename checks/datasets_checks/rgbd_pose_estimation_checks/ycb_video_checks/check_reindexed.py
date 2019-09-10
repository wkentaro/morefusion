#!/usr/bin/env python

import imgviz

import objslampp


class Images:

    def __init__(self):
        self._dataset = objslampp.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            'train'
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        dataset = self._dataset
        example = dataset[index]
        instance_id = dataset._ids[index]
        image_id = '/'.join(instance_id.split('/')[1:-1])
        frame = objslampp.datasets.YCBVideoDataset.get_frame(image_id)
        viz = imgviz.tile([
            example['rgb'],
            imgviz.depth2rgb(example['pcd'][:, :, 0]),
            imgviz.depth2rgb(example['pcd'][:, :, 1]),
            imgviz.depth2rgb(example['pcd'][:, :, 2]),
        ], border=(255, 255, 255))
        viz = imgviz.tile([frame['color'], viz], (1, 2))
        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
