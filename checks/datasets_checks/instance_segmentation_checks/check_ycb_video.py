#!/usr/bin/env python

import imgviz

import objslampp


class Images:

    dataset = objslampp.datasets.YCBVideoSyntheticInstanceSegmentationDataset()
    # dataset = objslampp.datasets.YCBVideoInstanceSegmentationDataset(
    #     split='train', sampling=15
    # )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image_id = self.dataset._ids[i]
        example = self.dataset[i]

        rgb = example['rgb']
        masks = example['masks']
        labels = example['labels']

        captions = objslampp.datasets.ycb_video.class_names[labels]
        viz = imgviz.instances2rgb(
            rgb, labels, masks=masks, captions=captions
        )
        viz = imgviz.draw.text_in_rectangle(
            viz, loc='lt', text=image_id, size=30, background=(0, 255, 0)
        )
        return imgviz.tile([rgb, viz], shape=(1, 2), border=(255, 255, 255))


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
