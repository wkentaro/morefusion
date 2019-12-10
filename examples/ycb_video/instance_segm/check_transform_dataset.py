#!/usr/bin/env python

import chainercv
import imgviz
import numpy as np

import morefusion

from train_multi import transform_dataset


class Images:

    def __init__(self, dataset, model):
        self._dataset = dataset
        self._model = model

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        rgb, bbox, label, mask = self._dataset[index]
        rgb += self._model.extractor.mean
        rgb = rgb.astype(np.uint8)
        rgb = rgb.transpose(1, 2, 0)

        class_ids = label + 1
        captions = [
            morefusion.datasets.ycb_video.class_names[c] for c in class_ids
        ]
        viz = imgviz.instances2rgb(
            rgb,
            labels=class_ids,
            masks=mask,
            bboxes=bbox,
            captions=captions,
        )
        viz = imgviz.tile([rgb, viz], shape=(1, 2), border=(255, 255, 255))
        return imgviz.resize(viz, width=1500)


model = chainercv.links.model.fpn.MaskRCNNFPNResNet50(
    n_fg_class=1, pretrained_model='imagenet'
)
dataset = \
    morefusion.datasets.MySyntheticYCB20190916InstanceSegmentationDataset(
        split='train', bg_composite=True
    )
# dataset = morefusion.datasets.YCBVideoInstanceSegmentationDataset(
#     split='train', sampling=15
# )
dataset = transform_dataset(dataset, model, train=True)
imgviz.io.pyglet_imshow(Images(dataset, model))
imgviz.io.pyglet_run()
