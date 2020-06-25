import collections
import json
import os.path as osp

import gdown

from ...ycb_video import YCBVideoDataset
from ...ycb_video import YCBVideoSyntheticDataset
from ..reindexed import RGBDPoseEstimationDatasetReIndexedBase


class YCBVideoRGBDPoseEstimationDatasetReIndexed(
    RGBDPoseEstimationDatasetReIndexedBase
):

    _root_dir = YCBVideoDataset._root_dir + ".reindexed.v2"

    def __init__(self, *args, **kwargs):
        if not self.root_dir.exists():
            self.download()
        super().__init__(*args, **kwargs)

    def download(self):
        gdown.cached_download(
            url="https://drive.google.com/uc?id=1l0ki7dX1WxcmV5Tfm41FPW-yk-wKUfne",  # NOQA
            path=self.root_dir + ".zip",
            postprocess=gdown.extractall,
        )

    def _get_ids(self):
        assert self.split in ["train", "trainreal", "syn", "val"]

        image_id_to_instance_ids = collections.defaultdict(list)
        with open(self.root_dir / "meta.json") as f:
            meta = json.load(f)
            for instance_id in meta:
                image_id = osp.dirname(instance_id)
                image_id_to_instance_ids[image_id].append(instance_id)
        image_id_to_instance_ids = dict(image_id_to_instance_ids)

        dataset = None
        if self.split == "val":
            sampling = 1
            dataset = YCBVideoDataset(split="keyframe")
        elif self.split in ["train", "trainreal"]:
            sampling = 8
            dataset = YCBVideoDataset(split="train")

        image_ids = []
        if dataset:
            image_ids = [
                f"data/{x}" for x in dataset.get_ids(sampling=sampling)
            ]

        if self.split in ["train", "syn"]:
            dataset_syn = YCBVideoSyntheticDataset()
            image_ids += [f"data_syn/{x}" for x in dataset_syn.get_ids()]

        ids = []
        for image_id in image_ids:
            instance_ids = image_id_to_instance_ids[image_id]
            for instance_id in instance_ids:
                class_id = meta[instance_id]["class_id"]
                if self._class_ids and class_id not in self._class_ids:
                    continue
                ids.append(instance_id)

        self._image_id_to_instance_ids = image_id_to_instance_ids
        return ids
