#!/usr/bin/env python

import morefusion

import imgviz


class Images:

    def __init__(self):
        self._dataset = morefusion.datasets.MyRealRGBDPoseEstimationDataset(  # NOQA
            root_dir='/home/wkentaro/data/datasets/wkentaro/morefusion/ycb_video/real_data/20190614_18'  # NOQA
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        dataset = self._dataset

        frame = dataset.get_frame(index)
        examples = dataset.get_example(index)

        vizs = []
        for example in examples:
            viz = imgviz.tile([
                example['rgb'],
                imgviz.depth2rgb(example['pcd'][:, :, 0]),
                imgviz.depth2rgb(example['pcd'][:, :, 1]),
                imgviz.depth2rgb(example['pcd'][:, :, 2]),
            ], border=(255, 255, 255))
            vizs.append(viz)
        viz = imgviz.tile(vizs)
        del vizs

        viz = imgviz.tile([frame['rgb'], viz], shape=(1, 2))
        viz = imgviz.resize(viz, width=1000)

        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
