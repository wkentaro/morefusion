import imgaug
import imgaug.augmenters as iaa
import imgviz
import numpy as np
import skimage.morphology

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    def _get_invalid_data(self):
        return dict(
            class_id=-1,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
            translation_rough=np.zeros((3,), dtype=np.float64),
        )

    def get_example(self, index):
        examples = self.get_examples(index)

        class_ids = [e['class_id'] for e in examples]

        if self._class_ids is None:
            class_id = np.random.choice(class_ids)
        else:
            options = set(self._class_ids) & set(class_ids)
            if options:
                class_id = np.random.choice(list(options))
            else:
                return self._get_invalid_data()
        instance_index = np.random.choice(np.where(class_ids == class_id)[0])

        return examples[instance_index]

    def _augment(self, rgb, depth):
        if 'rgb' in self._augmentation:
            rgb = self._augment_rgb(rgb)

        if 'depth' in self._augmentation:
            depth = self._augment_depth(depth)

        if 'segm' in self._augmentation:
            rgb, depth = self._augment_segmentation(rgb, depth)

        if 'occl' in self._augmentation:
            rgb, depth = self._augment_occlusion(rgb, depth)

        mask = ~np.isnan(depth)
        rgb[~mask] = 0
        depth[~mask] = np.nan
        return rgb, depth

    def _augment_rgb(self, rgb):
        augmenter = iaa.Sequential([
            iaa.ContrastNormalization(alpha=(0.8, 1.2)),
            iaa.WithColorspace(
                to_colorspace='HSV',
                from_colorspace='RGB',
                children=iaa.Sequential([
                    iaa.WithChannels(
                        (1, 2),
                        iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
                    ),
                    iaa.WithChannels(
                        (0,),
                        iaa.Multiply(mul=(0.95, 1.05), per_channel=True),
                    ),
                ]),
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.KeepSizeByResize(children=iaa.Resize((0.25, 1.0))),
        ])
        rgb = augmenter.augment_image(rgb)

        return rgb

    def _augment_depth(self, depth):
        random_state = imgaug.current_random_state()
        depth += random_state.normal(scale=0.01, size=depth.shape)
        return depth

    def _partial_binary_dilation(self, mask):
        random_state = imgaug.current_random_state()
        selem = skimage.morphology.disk(random_state.randint(1, 10))
        mask_dilated = skimage.morphology.binary_dilation(mask, selem)

        augmenter = iaa.CoarseDropout(p=(0.5, 0.9), size_px=(10, 30))
        mask_change = np.ones_like(mask)
        mask_change = augmenter.augment_image(
            mask_change.astype(np.uint8)
        ).astype(bool)

        mask_dst = mask.copy()
        mask_dst[mask_change] = mask_dilated[mask_change]
        return mask_dst

    def _augment_segmentation(self, rgb, depth):
        random_state = imgaug.current_random_state()

        mask = ~np.isnan(depth)
        rgb[~mask] = random_state.randint(0, 255, (np.sum(~mask), 3))
        # noise to background depth
        depth_copy = depth.copy()
        height, width = depth_copy.shape[:2]
        height_dst = int(round(height * 1.1))
        depth_copy = imgviz.resize(depth_copy, height=height_dst)
        y1 = int(round((height_dst - height) / 2))
        x1 = int(round((depth_copy.shape[1] - width) / 2))
        depth_copy = depth_copy[y1:y1 + height, x1:x1 + width]
        depth_copy += random_state.normal(scale=0.05, size=depth_copy.shape)
        depth[~mask] = depth_copy[~mask]

        # noise to mask
        mask = self._partial_binary_dilation(mask)
        augmenter = iaa.Sometimes(
            0.5,
            iaa.ElasticTransformation(alpha=(0, 70.0), sigma=8.0)
        )
        mask = augmenter.augment_image(mask.astype(float)) > 0.5

        rgb[~mask] = 0
        depth[~mask] = np.nan
        return rgb, depth

    def _augment_occlusion(self, rgb, depth):
        mask = ~np.isnan(depth)

        augmenter = iaa.Sequential([
            iaa.CoarseDropout(p=(0, 0.5), size_px=2, min_size=2),
            iaa.ElasticTransformation(alpha=(0, 50), sigma=5),
        ])

        mask_org = mask.copy()
        min_p = 0.5
        while True:
            mask = augmenter.augment_image(
                mask_org.astype(np.uint8)
            ).astype(bool)
            mask = mask & mask_org
            if (mask.sum() / mask_org.sum()) > min_p:
                break

        rgb[~mask] = 0
        depth[~mask] = np.nan
        return rgb, depth
