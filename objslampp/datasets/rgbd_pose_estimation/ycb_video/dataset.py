import numpy as np

from ...ycb_video import YCBVideoDataset
from ...ycb_video import YCBVideoModels
from ...ycb_video import YCBVideoSyntheticDataset
from ..base import RGBDPoseEstimationDatasetBase


class YCBVideoRGBDPoseEstimationDataset(RGBDPoseEstimationDatasetBase):

    _models = YCBVideoModels()
    _bounded_rate_minimal = 0.5

    def __init__(
        self,
        split,
        class_ids=None,
        sampling=None,
    ):
        if split != 'val':
            self._n_points_minimal = 50

        super().__init__(
            root_dir=YCBVideoDataset._root_dir,
            class_ids=class_ids,
        )

        assert isinstance(split, str)
        self._split = split
        self._sampling = sampling

        self._ids = self._get_ids()

    def get_voxel_pitch(self, class_id):
        return self._models.get_voxel_pitch(self._voxel_dim, class_id=class_id)

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

    def get_example(self, index):
        examples = super().get_example(index)

        if self.split == 'val':
            return examples

        examples_filtered = []
        for example in examples:
            diagonal = self._models.get_bbox_diagonal(example['class_id'])
            aabb_min = example['translation_true'] - (diagonal / 2.)
            aabb_max = aabb_min + diagonal

            nonnan = ~np.isnan(example['pcd']).any(axis=2)
            points = example['pcd'][nonnan]
            bounded = (
                (aabb_min <= points).all(axis=1) &
                (points < aabb_max).all(axis=1)
            )

            bounded_rate = bounded.sum() / len(points)
            if bounded_rate < self._bounded_rate_minimal:
                continue
            examples_filtered.append(example)

        return examples_filtered
