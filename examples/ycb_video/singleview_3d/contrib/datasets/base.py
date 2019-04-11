import imgaug
import imgaug.augmenters as iaa
import numpy as np

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    voxel_dim = 32
    _cache_pitch = {}

    _backgrounds = objslampp.datasets.YCBVideoDataset('train')
    _occlusions = _backgrounds

    def _get_invalid_data(self):
        return dict(
            class_id=-1,
            pitch=0.,
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            pcd=np.zeros((256, 256, 3), dtype=np.float64),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
        )

    def _get_pitch(self, class_id):
        if class_id in self._cache_pitch:
            return self._cache_pitch[class_id]

        models = objslampp.datasets.YCBVideoModels()
        cad_file = models.get_model(class_id=class_id)['textured_simple']
        bbox_diagonal = models.get_bbox_diagonal(mesh_file=cad_file)
        pitch = 1. * bbox_diagonal / self.voxel_dim
        pitch = pitch.astype(np.float32)

        self._cache_pitch[class_id] = pitch
        return pitch

    def _augment(self, rgb, depth):
        if 'rgb' in self._augmentation:
            rgb = self._augment_rgb(rgb)

        if 'depth' in self._augmentation:
            depth = self._augment_depth(depth)

        if 'occl' in self._augmentation:
            rgb, depth = self._augment_occlusion(rgb, depth)

        if 'segm' in self._augmentation:
            rgb, depth = self._augment_segmentation(rgb, depth)

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
        depth = depth.copy()
        random_state = imgaug.current_random_state()
        depth += random_state.normal(scale=0.01, size=depth.shape)
        return depth

    def _augment_segmentation(self, rgb, depth):
        rgb = rgb.copy()
        depth = depth.copy()

        random_state = imgaug.current_random_state()

        index = random_state.randint(0, len(self._backgrounds))
        background = self._backgrounds[index]
        rgb_bg = background['color'].copy()
        depth_bg = background['depth'].copy()

        mask = ~np.isnan(depth)
        bbox = objslampp.geometry.masks_to_bboxes(mask)
        by1, bx1, by2, bx2 = bbox.round().astype(int)
        bh, bw = by2 - by1, bx2 - bx1
        if bh * bw == 0:
            return rgb, depth

        cy, cx = random_state.uniform(
            (bh, bw), (rgb_bg.shape[0] - bh, rgb_bg.shape[1] - bw)
        ).round().astype(int)

        H, W = rgb_bg.shape[:2]
        y1 = np.clip(int(round(cy - bh / 2)), 0, H - 1)
        x1 = np.clip(int(round(cx - bw / 2)), 0, W - 1)
        y2 = np.clip(y1 + bh, 0, H - 1)
        x2 = np.clip(x1 + bw, 0, H - 1)

        by2 = by1 + (y2 - y1)
        bx2 = bx1 + (x2 - x1)

        mask_roi = mask[by1:by2, bx1:bx2]
        rgb_bg[y1:y2, x1:x2][mask_roi] = rgb[by1:by2, bx1:bx2][mask_roi]
        depth_bg[y1:y2, x1:x2][mask_roi] = depth[by1:by2, bx1:bx2][mask_roi]

        mask = mask_roi
        rgb = rgb_bg[y1:y2, x1:x2]
        depth = depth_bg[y1:y2, x1:x2]

        # randomly shift mask (like annotations in ycb_video)
        y0, x0 = random_state.normal(
            loc=10, scale=3.0, size=(2,)
        ).round().astype(int)
        y0, x0 = np.clip((y0, x0), 0, 20)
        mask = np.pad(mask, 10, mode='constant', constant_values=0)[
            y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]
        ]

        rgb[~mask] = 0
        depth[~mask] = np.nan
        return rgb, depth

    def _augment_occlusion(self, rgb, depth):
        random_state = imgaug.current_random_state()
        n_sample = random_state.randint(0, 3)
        min_ratio_occl = 0.3
        mask_old = ~np.isnan(depth)
        for _ in range(n_sample):
            rgb_new, depth_new = self._augment_occlusion_one(rgb, depth)
            mask_new = ~np.isnan(depth_new)
            ratio_occl_current = mask_new.sum() / mask_old.sum()
            if ratio_occl_current < min_ratio_occl:
                continue
            rgb, depth = rgb_new, depth_new
        return rgb, depth

    def _augment_occlusion_one(self, rgb, depth):
        rgb = rgb.copy()
        depth = depth.copy()

        mask = ~np.isnan(depth)

        random_state = imgaug.current_random_state()

        index = random_state.randint(0, len(self._backgrounds))
        occlusion = self._occlusions[index]
        rgb_occl = occlusion['color']
        labels = np.unique(occlusion['label'])
        labels = labels[labels > 0]
        mask_occl = occlusion['label'] == random_state.choice(labels)

        bbox = objslampp.geometry.masks_to_bboxes(mask_occl)
        by1, bx1, by2, bx2 = bbox.round().astype(int)
        bh, bw = by2 - by1, bx2 - bx1

        rgb_occl = rgb_occl[by1:by2, bx1:bx2]
        mask_occl = mask_occl[by1:by2, bx1:bx2]

        H, W = rgb.shape[:2]
        cy1, cx1, cy2, cx2 = objslampp.geometry.masks_to_bboxes(mask)
        cy, cx = random_state.uniform(
            (cy1, cx1), (cy2, cx2),
        ).round().astype(int)

        y1 = np.clip(int(round(cy - bh / 2)), 0, H - 1)
        x1 = np.clip(int(round(cx - bw / 2)), 0, W - 1)
        y2 = np.clip(y1 + bh, 0, H - 1)
        x2 = np.clip(x1 + bw, 0, W - 1)

        by1 = 0
        bx1 = 0
        by2 = by1 + (y2 - y1)
        bx2 = bx1 + (x2 - x1)

        mask_occl = mask_occl[by1:by2, bx1:bx2]
        rgb[y1:y2, x1:x2][mask_occl] = rgb_occl[by1:by2, bx1:bx2][mask_occl]
        mask[y1:y2, x1:x2][mask_occl] = 0

        rgb[~mask] = 0
        depth[~mask] = np.nan
        return rgb, depth
