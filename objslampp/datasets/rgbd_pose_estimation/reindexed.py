import cv2
import imgaug
import imgaug.augmenters as iaa
import imgviz
import numpy as np

from ... import geometry as geometry_module
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

    @staticmethod
    def _augment_mask(rgb, pcd):
        H, W = rgb.shape[:2]
        random_state = imgaug.random.get_global_rng()
        mask = ~np.isnan(pcd).any(axis=2)

        # shift xy
        case = random_state.choice(4)
        y1, x1, y2, x2 = geometry_module.masks_to_bboxes([mask])[0]
        if case == 0:
            y1 = random_state.uniform(0, H * 0.25)
        elif case == 1:
            y2 = H - random_state.uniform(0, H * 0.25)
        elif case == 2:
            x1 = random_state.uniform(0, W * 0.25)
        else:
            assert case == 3
            x2 = W - random_state.uniform(0, W * 0.25)
        y1, x1, y2, x2 = np.array([y1, x1, y2, x2]).round().astype(int)
        mask[:y1, :] = 0
        mask[y2:, :] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0

        # select blobs
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            contour_index = np.argmax(areas)
            mask_contour = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(
                mask_contour,
                contours,
                contourIdx=contour_index,
                thickness=-1,
                color=1,
            )
            n_contours = random_state.choice(len(contours))
            for contour_index in random_state.permutation(
                len(contours)
            )[:n_contours]:
                cv2.drawContours(
                    mask_contour,
                    contours,
                    contourIdx=contour_index,
                    thickness=-1,
                    color=1,
                )
            mask = mask_contour.astype(bool)

        rgb[~mask] = 0
        pcd[~mask] = np.nan

        bbox = geometry_module.masks_to_bboxes([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)

        rgb = rgb[y1:y2, x1:x2]
        pcd = pcd[y1:y2, x1:x2]
        rgb = imgviz.centerize(rgb, (H, W))
        pcd = imgviz.centerize(
            pcd, (H, W), cval=np.nan, interpolation='nearest'
        )
        return rgb, pcd

    @staticmethod
    def _augment_rgbd(rgb, pcd):
        rgb, pcd = RGBDPoseEstimationDatasetReIndexedBase._augment_mask(
            rgb, pcd
        )
        rgb = RGBDPoseEstimationDatasetReIndexedBase._augment_rgb(rgb)
        pcd = RGBDPoseEstimationDatasetReIndexedBase._augment_pcd(pcd)

        return rgb, pcd

    def get_example(self, index):
        id = self._ids[index]
        npz_file = self.root_dir / f'{id}.npz'
        example = dict(np.load(npz_file))
        if 'visibility' in example:
            example.pop('visibility')
        if self._augmentation:
            example['rgb'], example['pcd'] = self._augment_rgbd(
                example['rgb'], example['pcd'],
            )
        return example
