import chainercv
import numpy as np
import imgviz

from ... import geometry as geometry_module

from ..rgbd_pose_estimation import MySyntheticYCB20190916RGBDPoseEstimationDataset  # NOQA


class MySyntheticYCB20190916InstanceSegmentationDataset(
    MySyntheticYCB20190916RGBDPoseEstimationDataset
):

    def __init__(self, split, class_ids=None, add_background=False):
        super().__init__(split=split, class_ids=class_ids)
        self._random_state = np.random.mtrand._rand
        self._add_background = add_background
        if self._add_background:
            self._voc_dataset = chainercv.datasets.VOCBboxDataset()

    def get_frame(self, index):
        raise NotImplementedError

    def add_background(self, frame):
        index = self._random_state.randint(0, len(self._voc_dataset))
        bg = self._voc_dataset.get_example_by_keys(index, [0])[0]
        bg = bg.transpose(1, 2, 0)

        H_fg, W_fg = frame['rgb'].shape[:2]
        H_bg, W_bg = bg.shape[:2]

        H = max(H_fg, H_bg)
        W = max(W_fg, W_bg)

        scale = max(H / H_bg, W / W_bg)
        H = int(round(scale * H_bg))
        W = int(round(scale * W_bg))
        bg = imgviz.resize(bg, height=H, width=W, backend='opencv')

        y1 = self._random_state.randint(0, H - H_fg + 1)
        y2 = y1 + H_fg
        x1 = self._random_state.randint(0, W - W_fg + 1)
        x2 = x1 + W_fg
        bg_mask = frame['instance_label'] == 0
        frame['rgb'][bg_mask] = bg[y1:y2, x1:x2][bg_mask]

    def get_example(self, index):
        frame = super().get_frame(index)

        if self._add_background:
            self.add_background(frame)

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
