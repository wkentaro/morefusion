import imgaug
import imgaug.augmenters as iaa
import numpy as np

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    voxel_dim = 32
    _cache_pitch = {}

    _augmentation = {}
    _occlusions = objslampp.datasets.YCBVideoDataset('train')

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

    def get_example(self, index):
        examples = self.get_examples(index)

        class_ids = [e['class_id'] for e in examples]

        if self._class_ids:
            options = set(self._class_ids) & set(class_ids)
            if options:
                class_id = np.random.choice(list(options))
            else:
                return self._get_invalid_data()
        else:
            # None or []
            class_id = np.random.choice(class_ids)
        instance_index = np.random.choice(np.where(class_ids == class_id)[0])

        return examples[instance_index]

    def _augment(self, rgb, depth, mask):
        augmentation_all = {'rgb', 'depth', 'segm', 'occl'}
        assert augmentation_all.issuperset(set(self._augmentation))

        if 'rgb' in self._augmentation:
            rgb = self._augment_rgb(rgb)

        if 'depth' in self._augmentation:
            depth = self._augment_depth(depth)

        if 'segm' in self._augmentation:
            mask = self._augment_segmentation(mask)

        if 'occl' in self._augmentation:
            mask = self._augment_occlusion(mask)

        return rgb, depth, mask

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

    def _augment_segmentation(self, mask):
        H, W = mask.shape

        bbox = objslampp.geometry.masks_to_bboxes(mask)
        y1, x1, y2, x2 = bbox.round().astype(int)

        # randomly shift mask into inside
        random_state = imgaug.current_random_state()
        dydx = random_state.normal(loc=0, scale=10, size=(2,))
        dy, dx = dydx.round().astype(int)

        r, c = np.where(mask)
        r = np.clip(r + dy, 0, H - 1)
        c = np.clip(c + dx, 0, W - 1)
        mask_ = np.zeros_like(mask)
        mask_[r, c] = True

        mask = mask & mask_

        return mask

    def _augment_occlusion(self, mask):
        random_state = imgaug.current_random_state()
        n_sample = random_state.randint(0, 3)
        min_ratio_occl = 0.3
        mask_old = mask
        for _ in range(n_sample):
            mask_new = self._augment_occlusion_one(mask)
            ratio_occl_current = mask_new.sum() / mask_old.sum()
            if ratio_occl_current < min_ratio_occl:
                continue
            mask = mask_new
        return mask

    def _augment_occlusion_one(self, mask):
        mask = mask.copy()

        random_state = imgaug.current_random_state()

        index = random_state.randint(0, len(self._occlusions))
        occlusion = self._occlusions[index]
        labels = np.unique(occlusion['label'])
        labels = labels[labels > 0]
        mask_occl = occlusion['label'] == random_state.choice(labels)

        bbox = objslampp.geometry.masks_to_bboxes(mask_occl)
        by1, bx1, by2, bx2 = bbox.round().astype(int)
        bh, bw = by2 - by1, bx2 - bx1

        mask_occl = mask_occl[by1:by2, bx1:bx2]

        H, W = mask.shape[:2]
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
        mask[y1:y2, x1:x2][mask_occl] = 0

        return mask
