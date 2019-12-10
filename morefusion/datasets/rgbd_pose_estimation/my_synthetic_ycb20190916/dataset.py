import chainer

from ..my_synthetic import MySyntheticRGBDPoseEstimationDataset


class MySyntheticYCB20190916RGBDPoseEstimationDataset(
    MySyntheticRGBDPoseEstimationDataset
):

    def __init__(self, split, class_ids=None):
        root_dir = chainer.dataset.get_dataset_directory(
            'wkentaro/morefusion/ycb_video/synthetic_data/20190916_124002.877532.v2',  # NOQA
            create_directory=False,
        )
        super().__init__(
            root_dir=root_dir,
            class_ids=class_ids,
        )

        assert split in ['train', 'val']
        if split == 'train':
            self._ids = [i for i in self._ids if int(i.split('/')[0]) <= 1000]
            assert len(self._ids) == 15000
        else:
            assert split == 'val'
            self._ids = [i for i in self._ids if int(i.split('/')[0]) > 1000]
            assert len(self._ids) == 3000
