import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # NOQA
import pathlib

import chainer
import imgviz
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tf

import objslampp

from .base import DatasetBase


class CADOnlyDataset(DatasetBase):

    _cache_dir = chainer.dataset.get_dataset_directory(
        'wkentaro/objslampp/ycb_video/cad_only'
    )
    _cache_dir = pathlib.Path(_cache_dir)

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
        random_state = np.random.RandomState(index)

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

        cache_file = self._cache_dir / f'{class_id:04d}/{index:08d}.npz'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        load_cache = True
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                rgb = data['rgb']
                depth = data['depth']
                load_cache = False
            except Exception:
                load_cache = True
        if load_cache:
            scene = pyrender.Scene(bg_color=(0, 0, 0))
            scene.add(pyrender.Mesh.from_trimesh(cad), pose=T_cad2cam)
            camera_pose = objslampp.extra.trimesh.camera_transform()
            camera_node = pyrender.Node(
                camera=pyrender.PerspectiveCamera(
                    yfov=np.deg2rad(self.camera.fov[1]),
                    zfar=1000,
                    znear=0.01,
                ),
                matrix=camera_pose,
            )
            scene.add_node(camera_node)
            for node in create_raymond_lights(
                intensity=random_state.uniform(3, 7)
            ):
                scene.add_node(node, parent_node=camera_node)
            renderer = pyrender.OffscreenRenderer(
                self.camera.resolution[0], self.camera.resolution[1]
            )
            rgb, depth = renderer.render(scene)
            rgb = rgb.copy()
            depth = depth.copy()
            depth[depth == 0] = np.nan
            del renderer
            np.savez_compressed(cache_file, rgb=rgb, depth=depth)
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


def create_raymond_lights(intensity):
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(
                color=np.ones(3), intensity=intensity
            ),
            matrix=matrix
        ))

    return nodes


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
