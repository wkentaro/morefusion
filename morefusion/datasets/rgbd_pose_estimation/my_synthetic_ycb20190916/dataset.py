import gdown

from .... import utils as utils_module
from ..my_synthetic import MySyntheticRGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDataset(
    MySyntheticRGBDPoseEstimationDataset
):
    def __init__(self, split, class_ids=None, version=None):
        assert version in [None, 2]
        root_dir = utils_module.get_data_path(
            "wkentaro/morefusion/ycb_video/synthetic_data/20190916_124002.877532.v2",  # NOQA
        )
        super().__init__(
            root_dir=root_dir, class_ids=class_ids,
        )

        if not self.root_dir.exists():
            self.download()

        assert split in ["train", "val"]
        if split == "train":
            self._ids = [i for i in self._ids if int(i.split("/")[0]) <= 1000]
            assert len(self._ids) == 15000
        else:
            assert split == "val"
            self._ids = [i for i in self._ids if int(i.split("/")[0]) > 1000]
            assert len(self._ids) == 3000

    def download(self):
        if self.root_dir.endswith("v2"):
            return gdown.cached_download(
                url="https://drive.google.com/uc?id=1mwa1gQy8ibuc3Gc-6lgvt64VFmH5CW4q",  # NOQA
                path=self.root_dir + ".zip",
                md5="13f12f4c00ce4976217635632363c19a",
                postprocess=gdown.extractall,
            )
        else:
            raise NotImplementedError
