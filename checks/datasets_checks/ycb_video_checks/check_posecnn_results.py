#!/usr/bin/env python

import imgviz

import objslampp


class Images:

    def __init__(self):
        self._dataset = objslampp.datasets.YCBVideoPoseCNNResultsDataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        example = self._dataset[i]

        rgb = example['color']
        depth_viz = imgviz.depth2rgb(example['depth'])
        label_viz = imgviz.label2rgb(
            example['result']['labels'],
            label_names=objslampp.datasets.ycb_video.class_names,
        )

        viz = imgviz.tile(
            [rgb, depth_viz, label_viz],
            shape=(1, 3),
            border=(255, 255, 255),
        )
        viz = imgviz.resize(viz, width=1000)
        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
