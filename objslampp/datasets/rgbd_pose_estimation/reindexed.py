import imgaug
import imgaug.augmenters as iaa
import numpy as np

from ..base import DatasetBase


class RGBDPoseEstimationDatasetReIndexedBase(DatasetBase):

    def __init__(
        self,
        split,
        class_ids=None,
        augmentation=False,
    ):
        if not self.root_dir.exists():
            raise IOError(f'{self.root_dir} does not exist. ')

        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids

        assert isinstance(split, str)
        self._split = split

        self._augmentation = augmentation

        self._ids = self._get_ids()

    def get_indices_from_image_id(self, image_id):
        indices = []
        for id in self._image_id_to_instance_ids[image_id]:
            try:
                indices.append(self._ids.index(id))
            except ValueError:
                pass
        return indices

    @staticmethod
    def _augment_rgb(rgb):
        augmenter = iaa.Sequential([
            iaa.LinearContrast(alpha=(0.8, 1.2)),
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
