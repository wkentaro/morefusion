import imgaug
import imgaug.augmenters as iaa
import imgviz
import numpy as np
import skimage.morphology
import trimesh
import trimesh.transformations as tf

import objslampp

from .base import DatasetBase


class CADOnlyDataset(DatasetBase):

    def __init__(
        self,
        class_ids=None,
        n_sample_per_class=20000,
        augmentation={'rgb', 'depth', 'segm', 'occl'},
    ):
        self._models = objslampp.datasets.YCBVideoModels()

        if class_ids is None:
            class_ids = np.arange(1, self._models.n_class)
        self._class_ids = class_ids

        self._n_sample = n_sample_per_class * len(class_ids)

        augmentation_all = {'rgb', 'depth', 'segm', 'occl'}
        assert augmentation_all.issuperset(set(augmentation))
        self._augmentation = augmentation

    def __len__(self):
        return self._n_sample

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

    def _augment_depth(self, depth):
        random_state = imgaug.current_random_state()
        depth += random_state.normal(scale=0.01, size=depth.shape)
        return depth

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

    @property
    def camera(self):
        return trimesh.scene.Camera(resolution=(320, 320), fov=(60, 60))

    def get_frame(self, index):
        assert self._rgb[0] == index
        return dict(
            rgb=self._rgb[1],
            intrinsic_matrix=self.camera.K,
        )

    def get_example(self, index):
        random_state = imgaug.current_random_state()

        class_id = random_state.choice(self._class_ids)
        cad_file = self._models.get_cad_model(class_id=class_id)

        eye = objslampp.geometry.points_from_angles(
            distance=0.3,
            elevation=random_state.uniform(-90, 90),
            azimuth=random_state.uniform(-180, 180),
        )
        cad = trimesh.load(str(cad_file))
        target = random_state.uniform(- cad.extents / 2, cad.extents / 2, (3,))
        up = random_state.uniform(-1, 1, (3,))
        up /= np.linalg.norm(up)
        T_cam2cad = objslampp.geometry.look_at(
            eye=eye, target=target, up=up
        )
        T_cad2cam = np.linalg.inv(T_cam2cad)
        rgb, depth, _ = objslampp.extra.pybullet.render_cad(
            cad_file,
            T_cad2cam,
            fovy=self.camera.fov[1],
            height=self.camera.resolution[1],
            width=self.camera.resolution[0],
        )
        mask = ~np.isnan(depth)
        rgb[~mask] = 0

        if self._augmentation:
            rgb, depth = self._augment(rgb, depth)

        self._rgb = index, rgb

        K = self.camera.K
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        mask = ~np.isnan(depth)
        translation_rough = np.nanmean(pcd[mask], axis=0)

        bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)
        rgb = rgb[y1:y2, x1:x2]
        pcd = pcd[y1:y2, x1:x2]

        rgb = imgviz.centerize(rgb, (256, 256))
        pcd = imgviz.centerize(pcd, (256, 256), np.nan)

        quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
        translation_true = tf.translation_from_matrix(T_cad2cam)

        return dict(
            class_id=class_id,
            rgb=rgb,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            translation_rough=translation_rough,
        )


if __name__ == '__main__':
    dataset = CADOnlyDataset(class_ids=[2])
    print(f'dataset_size: {len(dataset)}')

    def images():
        for i in range(0, len(dataset)):
            example = dataset[i]
            print(f'index: {i:08d}')
            print(f"class_id: {example['class_id']}")
            print(f"quaternion_true: {example['quaternion_true']}")
            print(f"translation_true: {example['translation_true']}")
            print(f"translation_rough: {example['translation_rough']}")
            if example['class_id'] > 0:
                yield imgviz.tile(
                    [dataset.get_frame(i)['rgb'], example['rgb']], (1, 2)
                )

    imgviz.io.pyglet_imshow(images())
    imgviz.io.pyglet_run()
