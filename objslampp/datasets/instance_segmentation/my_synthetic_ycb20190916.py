import numpy as np

from ... import geometry as geometry_module

from ..rgbd_pose_estimation import MySyntheticYCB20190916RGBDPoseEstimationDataset  # NOQA


class MySyntheticYCB20190916InstanceSegmentationDataset(
    MySyntheticYCB20190916RGBDPoseEstimationDataset
):

    def get_frame(self, index):
        raise NotImplementedError

    def get_example(self, index):
        frame = super().get_frame(index)

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
