import os.path as osp
import collections
import json

from ..reindexed import RGBDPoseEstimationDatasetReIndexedBase
from ...ycb_video import YCBVideoDataset
from ...ycb_video import YCBVideoSyntheticDataset


class YCBVideoRGBDPoseEstimationDatasetReIndexed(
    RGBDPoseEstimationDatasetReIndexedBase
):

    _root_dir = YCBVideoDataset._root_dir + '.reindexed'

    def _get_ids(self):
        assert self.split in ['train', 'trainreal', 'syn', 'val']

        image_id_to_instance_ids = collections.defaultdict(list)
        with open(self.root_dir / 'id_to_class_id.json') as f:
            instance_id_to_class_id = json.load(f)
            for instance_id, class_id in instance_id_to_class_id.items():
                image_id = osp.dirname(instance_id)
                image_id_to_instance_ids[image_id].append(instance_id)
        image_id_to_instance_ids = dict(image_id_to_instance_ids)

        if self.split == 'val':
            sampling = 1
            dataset = YCBVideoDataset(split='keyframe')
        elif self.split in ['train', 'trainreal']:
            sampling = 8
            dataset = YCBVideoDataset(split='train')
        image_ids = [f'data/{x}' for x in dataset.get_ids(sampling=sampling)]

        if self.split in ['train', 'syn']:
            dataset_syn = YCBVideoSyntheticDataset()
            image_ids += [f'data_syn/{x}' for x in dataset_syn.get_ids()]

        ids = []
        for image_id in image_ids:
            instance_ids = image_id_to_instance_ids[image_id]
            for instance_id in instance_ids:
                class_id = instance_id_to_class_id[instance_id]
                if self._class_ids and class_id not in self._class_ids:
                    continue
                ids.append(instance_id)

        self._image_id_to_instance_ids = image_id_to_instance_ids
        return ids
