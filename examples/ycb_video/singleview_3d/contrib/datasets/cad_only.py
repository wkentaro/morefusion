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
        augmentation={'rgb', 'depth', 'segm', 'occl'}
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
        cad_file = self._models.get_cad_model(class_id=class_id)

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
        rgb[~mask] = 0
        depth[~mask] = np.nan

        # keep rgb and index for get_frame
        self._rgb = index, rgb

        # augment
        if self._augmentation:
            rgb, depth = self._augment(rgb, depth)
            mask = ~np.isnan(depth)

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
        pcd = imgviz.centerize(pcd, (256, 256), np.nan)

        quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
        translation_true = tf.translation_from_matrix(T_cad2cam)

        return dict(
            class_id=class_id,
            pitch=self._get_pitch(class_id=class_id),
            rgb=rgb,
            pcd=pcd,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )


if __name__ == '__main__':
    dataset = CADOnlyDataset(class_ids=[2])
    print(f'dataset_size: {len(dataset)}')

    if 0:
        example = dataset[0]
        rgb = example['rgb']
        pcd = example['pcd']
        mask = ~np.isnan(pcd).any(axis=2)
        pcd = trimesh.PointCloud(vertices=pcd[mask], colors=rgb[mask])
        objslampp.extra.trimesh.show_with_rotation(pcd.scene())
        del example, rgb, pcd, mask

    def images():
        for i in range(0, len(dataset)):
            example = dataset[i]
            print(f"class_id: {example['class_id']}")
            print(f"pitch: {example['pitch']}")
            print(f"quaternion_true: {example['quaternion_true']}")
            print(f"translation_true: {example['translation_true']}")
            if example['class_id'] > 0:
                viz = imgviz.tile([
                    example['rgb'],
                    imgviz.depth2rgb(example['pcd'][:, :, 0]),
                    imgviz.depth2rgb(example['pcd'][:, :, 1]),
                    imgviz.depth2rgb(example['pcd'][:, :, 2]),
                ], (1, 4), border=(255, 255, 255))
                yield viz

    imgviz.io.pyglet_imshow(images())
    imgviz.io.pyglet_run()
