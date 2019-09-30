import numpy as np

from ... import geometry as geometry_module
from ..ycb_video import YCBVideoDataset
from ..ycb_video import YCBVideoSyntheticDataset
from .voc_background_composite import VOCBackgroundComposite


def _ycb_video_to_instance_segmentation(example):
    rgb = example['color']
    lbl = example['label']
    meta = example['meta']

    labels = meta['cls_indexes'].astype(np.int32)
    masks = lbl[None, :, :] == labels[:, None, None]

    keep = masks.sum(axis=(1, 2)) > 0
    labels = labels[keep]
    masks = masks[keep]

    bboxes = geometry_module.masks_to_bboxes(masks)

    return dict(
        rgb=rgb,
        bboxes=bboxes,
        labels=labels,
        masks=masks,
    )


class YCBVideoSyntheticInstanceSegmentationDataset(YCBVideoSyntheticDataset):

    """YCBVideoSyntheticInstanceSegmentationDataset()

    Instance segmentation dataset of YCBVideoSyntheticDataset.

    .. seealso::
        See :class:`objslampp.datasets.YCBVideoSyntheticDataset`.

    """

    def __init__(self, bg_composite=False):
        self._bg_composite = None
        if bg_composite:
            self._bg_composite = VOCBackgroundComposite(bg_instance_ids=[0])
        super().__init__()

    def get_example(self, i):
        example = super().get_example(i)

        if self._bg_composite:
            example['color'] = self._bg_composite(
                example['color'], example['label']
            )

        return _ycb_video_to_instance_segmentation(example)


class YCBVideoInstanceSegmentationDataset(YCBVideoDataset):

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
