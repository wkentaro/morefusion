import os.path as osp
import collections
import json

import numpy as np

from ...base import DatasetBase
from .dataset import MySyntheticYCB20190916RGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(DatasetBase):

    def __init__(
        self,
        split,
        class_ids=None,
    ):
        self._root_dir = MySyntheticYCB20190916RGBDPoseEstimationDataset('train').root_dir + '.reindexed'  # NOQA

        if not self.root_dir.exists():
            raise IOError(
                f'{self.root_dir} does not exist. '
                'Please run following: python -m '
                'objslampp.datasets.rgbd_pose_estimation.my_synthetic_ycb20190916.reindex'  # NOQA
            )

        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids

        assert isinstance(split, str)
        self._split = split

        self._ids = self._get_ids()

    def _get_ids(self):
        assert self.split in ['train', 'val']

        image_id_to_instance_ids = collections.defaultdict(list)
        with open(self.root_dir / 'id_to_class_id.json') as f:
            instance_id_to_class_id = json.load(f)
            for instance_id, class_id in instance_id_to_class_id.items():
                image_id = osp.dirname(instance_id)
                image_id_to_instance_ids[image_id].append(instance_id)
        image_id_to_instance_ids = dict(image_id_to_instance_ids)

        ids = []
        for image_id, instance_ids in sorted(image_id_to_instance_ids.items()):
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
