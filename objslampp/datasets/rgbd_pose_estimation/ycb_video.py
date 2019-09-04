import numpy as np

from ..ycb_video import YCBVideoDataset
from ..ycb_video import YCBVideoSyntheticDataset
from .base import RGBDPoseEstimationDatasetBase


class YCBVideoRGBDPoseEstimationDataset(RGBDPoseEstimationDatasetBase):

    _root_dir = YCBVideoDataset._root_dir

    def __init__(
        self,
        split,
        class_ids=None,
        sampling=None,
    ):
        super().__init__(class_ids=class_ids)

        assert isinstance(split, str)
        self._split = split
        self._sampling = sampling

        self._ids = self._get_ids()

    def _get_ids(self):
        assert self.split in ['train', 'syn', 'val']

        if self.split == 'val':
            sampling = 1 if self._sampling is None else self._sampling
            dataset = YCBVideoDataset(split='keyframe')
        elif self.split == 'train':
            sampling = 8 if self._sampling is None else self._sampling
            dataset = YCBVideoDataset(split='train')

        ids = [(dataset, x) for x in dataset.get_ids(sampling=sampling)]

        if self.split in ['train', 'syn']:
            dataset = YCBVideoSyntheticDataset()
            ids += [(dataset, x) for x in dataset.get_ids()]

        return tuple(ids)

    def get_frame(self, index):
        dataset, image_id = self._ids[index]
        frame = dataset.get_frame(image_id)
        class_ids = frame['meta']['cls_indexes'].astype(np.int32)
        instance_ids = class_ids.copy()
        if 'rotation_translation_matrix' in frame['meta']:
            T_cam2world = frame['meta']['rotation_translation_matrix']
            T_cam2world = np.r_[T_cam2world, [[0, 0, 0, 1]]].astype(float)
        else:
            T_cam2world = np.eye(4, dtype=float)
        n_instance = len(instance_ids)
        Ts_cad2cam = np.zeros((n_instance, 4, 4), dtype=float)
        for i in range(n_instance):
            T_cad2cam = frame['meta']['poses'][:, :, i]
            T_cad2cam = np.r_[T_cad2cam, [[0, 0, 0, 1]]]
            Ts_cad2cam[i] = T_cad2cam
        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=frame['color'],
            depth=frame['depth'],
            instance_label=frame['label'],
            intrinsic_matrix=frame['meta']['intrinsic_matrix'],
            T_cam2world=T_cam2world,
            Ts_cad2cam=Ts_cad2cam,
            cad_files={},
        )
