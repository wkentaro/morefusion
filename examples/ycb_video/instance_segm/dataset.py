import numpy as np

import objslampp


def _ycb_video_to_instance_segmentation(example):
    rgb = example['color']
    lbl = example['label']
    meta = example['meta']

    labels = meta['cls_indexes'].astype(np.int32)
    masks = lbl[None, :, :] == labels[:, None, None]

    keep = masks.sum(axis=(1, 2)) > 0
    labels = labels[keep]
    masks = masks[keep]

    bboxes = objslampp.geometry.masks_to_bboxes(masks)

    return dict(
        rgb=rgb,
        bboxes=bboxes,
        labels=labels,
        masks=masks,
    )


class YCBVideoSyntheticInstanceSegmentationDataset(
    objslampp.datasets.YCBVideoSyntheticDataset
):

    """YCBVideoSyntheticInstanceSegmentationDataset()

    Instance segmentation dataset of YCBVideoSyntheticDataset.

    .. seealso::
        See :class:`objslampp.datasets.YCBVideoSyntheticDataset`.

    """

    def get_example(self, i):
        example = super().get_example(i)
        return _ycb_video_to_instance_segmentation(example)


class YCBVideoInstanceSegmentationDataset(objslampp.datasets.YCBVideoDataset):

    """YCBVideoInstanceSegmentationDataset(split: str, sampling: int = 1)

    Instance segmentation dataset of YCBVideoDataset.

    Parameters
    ----------
    split: str
        Split of this dataset.
        (choices: ('train', 'val', 'trainval', 'keyframe'))
    sampling: int
        Sampling step of the video frames. (default: 1)

    Properties
    ----------
    root_dir: path.Path
        Root directory of this dataset.

    .. seealso::
        See :class:`objslampp.datasets.YCBVideoDataset`.

    """

    def get_example(self, i):
        example = super().get_example(i)
        return _ycb_video_to_instance_segmentation(example)


if __name__ == '__main__':
    import imgviz

    class Images:

        dataset = YCBVideoSyntheticInstanceSegmentationDataset()
        # dataset = YCBVideoInstanceSegmentationDataset(
        #     split='train', sampling=15)

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
            return viz

    imgviz.io.pyglet_imshow(Images())
    imgviz.io.pyglet_run()
