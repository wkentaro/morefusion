from .... import utils as utils_module
from ..my_synthetic import MySyntheticRGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDataset(
    MySyntheticRGBDPoseEstimationDataset
):
    def __init__(self, split, class_ids=None, version=None):
        if version is None:
            version = 2
        if version == 2:
            root_dir = utils_module.get_data_path(
                "wkentaro/morefusion/ycb_video/synthetic_data/20190916_124002.877532.v2",  # NOQA
            )
        else:
            assert version == 1
            root_dir = utils_module.get_data_path(
                "wkentaro/morefusion/ycb_video/synthetic_data/20190916_124002.877532",  # NOQA
            )
        super().__init__(
            root_dir=root_dir, class_ids=class_ids,
        )

        assert split in ["train", "val"]
        if split == "train":
            self._ids = [i for i in self._ids if int(i.split("/")[0]) <= 1000]
            assert len(self._ids) == 15000
        else:
            assert split == "val"
            self._ids = [i for i in self._ids if int(i.split("/")[0]) > 1000]
            assert len(self._ids) == 3000
