import numpy as np

from ... import geometry as geometry_module
from ..rgbd_pose_estimation import MySyntheticYCB20190916RGBDPoseEstimationDataset  # NOQA
from .voc_background_composite import VOCBackgroundComposite


class MySyntheticYCB20190916InstanceSegmentationDataset(
    MySyntheticYCB20190916RGBDPoseEstimationDataset
):

    def __init__(self, split, class_ids=None, bg_composite=False):
        super().__init__(split=split, class_ids=class_ids)
        self._random_state = np.random.mtrand._rand
        self._bg_composite = None
        if bg_composite:
            self._bg_composite = VOCBackgroundComposite(bg_instance_ids=[0])

    def get_frame(self, index):
        raise NotImplementedError

    def get_example(self, index):
        frame = super().get_frame(index)

        if self._bg_composite:
            frame['rgb'] = self._bg_composite(
                frame['rgb'], frame['instance_label']
            )

        rgb = frame['rgb']
        labels = frame['class_ids']

        masks = []
        for instance_id in frame['instance_ids']:
            mask = frame['instance_label'] == instance_id
            masks.append(mask)
        masks = np.stack(masks).astype(bool)

        keep = labels > 0
        labels = labels[keep]
        masks = masks[keep]

        keep = masks.sum(axis=(1, 2)) > 0
        labels = labels[keep]
        masks = masks[keep]

        bboxes = geometry_module.masks_to_bboxes(masks)

        return dict(
            rgb=rgb,
            bboxes=bboxes,
            labels=labels,  # class_ids, bg is 0
            masks=masks,
        )
