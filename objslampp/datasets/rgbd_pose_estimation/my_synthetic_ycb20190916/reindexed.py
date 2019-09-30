import os.path as osp
import collections
import json

import imgaug
import imgaug.augmenters as iaa
import numpy as np

from ...base import DatasetBase
from .dataset import MySyntheticYCB20190916RGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(DatasetBase):

    def __init__(
        self,
        split: str,
        class_ids=None,
        augmentation: bool = False,
    ):
        self._root_dir = MySyntheticYCB20190916RGBDPoseEstimationDataset('train').root_dir + '.reindexed'  # NOQA
        if not self.root_dir.exists():
            raise IOError(
                f'{self.root_dir} does not exist. '
                'Please run following: python -m '
                'objslampp.datasets.rgbd_pose_estimation.my_synthetic_ycb20190916.reindex'  # NOQA
            )

        self._split = split
        self._class_ids = class_ids if class_ids is None else tuple(class_ids)
        self._augmentation = augmentation

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

    @staticmethod
    def _augment_rgb(rgb):
        augmenter = iaa.Sequential([
            iaa.ContrastNormalization(alpha=(0.8, 1.2)),
            iaa.WithColorspace(
                to_colorspace='HSV',
                from_colorspace='RGB',
                children=iaa.Sequential([
                    # SV
                    iaa.WithChannels(
                        (1, 2),
                        iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
                    ),
                    # H
                    iaa.WithChannels(
                        (0,),
                        iaa.Multiply(mul=(0.95, 1.05), per_channel=True),
                    ),
                ]),
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.KeepSizeByResize(children=iaa.Resize((0.25, 1.0))),
        ])
        return augmenter.augment_image(rgb)

    @staticmethod
    def _augment_pcd(pcd):
        random_state = imgaug.random.get_global_rng()
        dropout = random_state.binomial(1, 0.05, size=pcd.shape[:2])
        pcd[dropout == 1] = np.nan
        pcd += random_state.normal(0, 0.003, size=pcd.shape)
        return pcd

    def get_example(self, index):
        id = self._ids[index]
        npz_file = self.root_dir / f'{id}.npz'
        example = dict(np.load(npz_file))
        if self._augmentation:
            example['rgb'] = self._augment_rgb(example['rgb'])
            example['pcd'] = self._augment_pcd(example['pcd'])
        return example
