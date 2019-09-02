import chainer
import numpy as np
import path

import objslampp

from .base import DatasetBase


class YCBVideoDataset(DatasetBase):

    _root_dir = objslampp.datasets.YCBVideoDataset._root_dir
    _cache_dir = chainer.dataset.get_dataset_directory(
        'wkentaro/objslampp/ycb_video/singleview_3d/ycb_video/cache'
    )
    _cache_dir = path.Path(_cache_dir)

    def __init__(
        self,
        split,
        class_ids=None,
        sampling=None,
        num_syn=1.0,
    ):
        super().__init__(
            class_ids=class_ids,
        )

        assert isinstance(split, str)
        self._split = split
        self._sampling = sampling
        assert 0 < num_syn <= 1
        self._num_syn = num_syn

        self._dataset = None
        self._dataset_syn = None
        self._ids = self._get_ids()

    def _get_ids(self):
        assert self.split in ['train', 'syn', 'val']

        if self.split == 'val':
            sampling = 1 if self._sampling is None else self._sampling
            self._dataset = objslampp.datasets.YCBVideoDataset(
                split='keyframe'
            )
            ids = self._dataset.get_ids(sampling=sampling)
        elif self.split == 'train':
            sampling = 8 if self._sampling is None else self._sampling
            self._dataset = objslampp.datasets.YCBVideoDataset(
                split='train'
            )
            ids = self._dataset.get_ids(sampling=sampling)
        elif self.split == 'syn':
            ids = []

        ids = [(True, x) for x in ids]

        if self.split in ['train', 'syn']:
            self._dataset_syn = objslampp.datasets.YCBVideoSyntheticDataset()
            ids_syn = self._dataset_syn.get_ids()
            ids_syn = [(False, x) for x in ids_syn]
            num_syn = int(round(self._num_syn * len(ids_syn)))
            ids += ids_syn[:num_syn]

        return tuple(ids)

    def get_frame(self, index):
        is_real, image_id = self._ids[index]
        if is_real:
            dataset = self._dataset
        else:
            dataset = self._dataset_syn
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
