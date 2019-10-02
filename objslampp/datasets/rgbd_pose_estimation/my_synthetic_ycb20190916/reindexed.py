import os.path as osp
import collections
import json

from ..reindexed import RGBDPoseEstimationDatasetReIndexedBase
from .dataset import MySyntheticYCB20190916RGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(
    RGBDPoseEstimationDatasetReIndexedBase
):

    def __init__(
        self,
        split: str,
        class_ids=None,
        augmentation: bool = False,
    ):
        self._root_dir = MySyntheticYCB20190916RGBDPoseEstimationDataset(split).root_dir + '.reindexed'  # NOQA
        super().__init__(
            split=split,
            class_ids=class_ids,
            augmentation=augmentation,
        )

    def _get_ids(self):
        assert self.split in ['train', 'val']

        image_id_to_instance_ids = collections.defaultdict(list)
        with open(self.root_dir / 'id_to_class_id.json') as f:
            instance_id_to_class_id = json.load(f)
            for instance_id, class_id in instance_id_to_class_id.items():
                image_id = osp.dirname(instance_id)
                image_id_to_instance_ids[image_id].append(instance_id)
        image_id_to_instance_ids = dict(image_id_to_instance_ids)

        image_ids = MySyntheticYCB20190916RGBDPoseEstimationDataset(
            self.split
        )._ids

        ids = []
        for image_id in image_ids:
            if image_id not in image_id_to_instance_ids:
                continue
            instance_ids = image_id_to_instance_ids[image_id]
            for instance_id in instance_ids:
                class_id = instance_id_to_class_id[instance_id]
                if self._class_ids and class_id not in self._class_ids:
                    continue
                ids.append(instance_id)

        self._image_id_to_instance_ids = image_id_to_instance_ids
        return ids
