import imgaug
import imgviz
import numpy as np
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
        self._augmentation = augmentation

        self._n_sample = n_sample_per_class * len(class_ids)

    def __len__(self):
        return self._n_sample

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

        # prepare class_id and cad_file
        class_id = random_state.choice(self._class_ids)
        cad_file = self._models.get_cad_file(class_id=class_id)

        # get frame
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

        # keep rgb and index for get_frame
        self._rgb = index, rgb

        # augment
        if self._augmentation:
            rgb, depth, mask = self._augment(rgb, depth, mask)

        # masking
        rgb[~mask] = 0
        depth[~mask] = np.nan

        # get point cloud
        K = self.camera.K
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        # crop
        bbox = objslampp.geometry.masks_to_bboxes(mask)
        y1, x1, y2, x2 = bbox.round().astype(int)
        if (y2 - y1) * (x2 - x1) == 0:
            return self._get_invalid_data()
        rgb = rgb[y1:y2, x1:x2]
        pcd = pcd[y1:y2, x1:x2]

        # finalize
        rgb = imgviz.centerize(rgb, (256, 256))

        quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
        translation_true = tf.translation_from_matrix(T_cad2cam)
        translation_rough = np.nanmean(pcd, axis=(0, 1))

        return dict(
            class_id=class_id,
            rgb=rgb,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            translation_rough=translation_rough,
        )


if __name__ == '__main__':
    dataset = CADOnlyDataset(
        class_ids=[2],
        augmentation={'rgb', 'depth', 'segm', 'occl'},
    )
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
