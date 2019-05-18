#!/usr/bin/env python

from chainercv.links.model.fpn import MaskRCNNFPNResNet50
import imgviz
import numpy as np

import objslampp

from dataset import YCBVideoInstanceSegmentationDataset


class Images:

    dataset = YCBVideoInstanceSegmentationDataset(
        split='keyframe', sampling=10)
    class_names = objslampp.datasets.ycb_video.class_names

    model = MaskRCNNFPNResNet50(
        n_fg_class=len(class_names[1:]),
        pretrained_model='logs/2019-05-18_07-17-29/model_iter_best',
    )
    model.to_gpu()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image_id = self.dataset._ids[i]

        rgb = self.dataset[i]['rgb']

        with objslampp.utils.timer():
            masks, labels, confs = self.model.predict(
                [rgb.astype(np.float32).transpose(2, 0, 1)]
            )
        masks = masks[0]
        labels = labels[0]
        confs = confs[0]

        keep = masks.sum(axis=(1, 2)) > 0
        masks = masks[keep]
        labels = labels[keep]
        confs = confs[keep]

        captions = [
            f'{self.class_names[label]}: {conf:.1%}'
            for label, conf in zip(labels, confs)
        ]
        for caption in captions:
            print(caption)
        viz = imgviz.instances.instances2rgb(
            image=rgb,
            masks=masks,
            labels=labels,
            captions=captions,
            font_size=15,
        )
        viz = imgviz.tile([rgb, viz], (1, 2), border=(0, 0, 0))
        viz = imgviz.draw.text_in_rectangle(
            viz, loc='lt', text=image_id, size=25, background=(0, 255, 0)
        )
        return viz


imgviz.io.pyglet_imshow(Images(), interval=1 / 10)
imgviz.io.pyglet_run()
