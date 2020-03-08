#!/usr/bin/env python

import os.path as osp

import imgviz

import morefusion


class Images:
    def __init__(self):
        self._dataset_parent = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDataset(  # NOQA
            "train"
        )
        self._dataset = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(  # NOQA
            "train", augmentation=True
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        example = self._dataset[index]
        instance_id = self._dataset._ids[index]

        image_id = osp.dirname(instance_id)
        index_parent = self._dataset_parent._ids.index(image_id)
        frame = self._dataset_parent.get_frame(index_parent)

        print(instance_id)
        viz = imgviz.tile(
            [
                example["rgb"],
                imgviz.depth2rgb(example["pcd"][:, :, 0]),
                imgviz.depth2rgb(example["pcd"][:, :, 1]),
                imgviz.depth2rgb(example["pcd"][:, :, 2]),
            ],
            border=(255, 255, 255),
        )
        viz = imgviz.tile([frame["rgb"], viz], shape=(1, 2))
        return viz


imgviz.io.pyglet_imshow(Images())
imgviz.io.pyglet_run()
