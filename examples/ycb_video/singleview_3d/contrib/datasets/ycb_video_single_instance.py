import os.path as osp
import collections
import json

import chainer
import numpy as np

import objslampp


class YCBVideoSingleInstanceDataset(objslampp.datasets.DatasetBase):

    _root_dir = chainer.dataset.get_dataset_directory(
        'wkentaro/objslampp/ycb_video/single_instance_dataset'
    )

    def __init__(
        self,
        split,
        sampling=None,
        class_ids=None,
        num_syn=1.0,
    ):
        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids

        assert isinstance(split, str)
        self._split = split
        self._sampling = sampling
        assert 0 < num_syn <= 1
        self._num_syn = num_syn

        self._ids = self._get_ids()

    def _get_ids(self):
        assert self.split in ['train', 'syn', 'val']

        image_id_to_instance_ids = collections.defaultdict(list)
        with open(self.root_dir / 'id_to_class_id.json') as f:
            instance_id_to_class_id = json.load(f)
            for instance_id, class_id in instance_id_to_class_id.items():
                image_id = osp.dirname(instance_id)
                image_id_to_instance_ids[image_id].append(instance_id)
        image_id_to_instance_ids = dict(image_id_to_instance_ids)

        if self.split == 'val':
            sampling = 1 if self._sampling is None else self._sampling
            dataset = objslampp.datasets.YCBVideoDataset(
                split='keyframe'
            )
        elif self.split == 'train':
            sampling = 8 if self._sampling is None else self._sampling
            dataset = objslampp.datasets.YCBVideoDataset(
                split='train'
            )
        image_ids = [f'data/{x}' for x in dataset.get_ids(sampling=sampling)]

        if self.split in ['train', 'syn']:
            dataset_syn = objslampp.datasets.YCBVideoSyntheticDataset()
            image_ids_syn = [f'data_syn/{x}' for x in dataset_syn.get_ids()]
            num_syn = int(round(self._num_syn * len(image_ids_syn)))
            image_ids += image_ids_syn[:num_syn]

        ids = []
        for image_id in image_ids:
            instance_ids = image_id_to_instance_ids[image_id]
            for instance_id in instance_ids:
                class_id = instance_id_to_class_id[instance_id]
                if self._class_ids and class_id not in self._class_ids:
                    continue
                ids.append(instance_id)

        return ids

    def get_example(self, index):
        id = self._ids[index]
        npz_file = self.root_dir / f'{id}.npz'
        return dict(np.load(npz_file))
