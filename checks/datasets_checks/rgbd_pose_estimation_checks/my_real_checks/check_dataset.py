#!/usr/bin/env python

import gdown
import imgviz

import morefusion


class Images:

    def __init__(self):
        root_dir = morefusion.utils.get_data_path(
            'wkentaro/morefusion/ycb_video/real_data/20191212_163242.566559922'
        )
        gdown.cached_download(
            url='https://drive.google.com/uc?id=1llWN7MOLzJZnaRDD4XGSmRWAFBtP3P9z',  # NOQA
            md5='a773bb947377811b2b66ab9bc17f4d8d',
            path=root_dir + '.zip',
        )
        self._dataset = morefusion.datasets.MyRealRGBDPoseEstimationDataset(
            root_dir=root_dir,
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
