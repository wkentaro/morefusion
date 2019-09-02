#!/usr/bin/env python

import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np
import trimesh
import trimesh.transformations as tf

import objslampp


class NearestNeighborICP(chainer.Link):

    def __init__(self, quaternion_init=None, translation_init=None):
        super().__init__()

        if quaternion_init is None:
            quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
        if translation_init is None:
            translation_init = np.array([0, 0, 0], dtype=np.float32)

        with self.init_scope():
            self.quaternion = chainer.Parameter(initializer=quaternion_init)
            self.translation = chainer.Parameter(initializer=translation_init)

    @property
    def T(self):
        return objslampp.functions.transformation_matrix(
            self.quaternion, self.translation
        )

    def forward(self, source, target):
        # source: from cad
        # target: from depth

        source = objslampp.functions.transform_points(source, self.T[None])[0]

        dists = F.sum(
            (source[None, :, :] - target[:, None, :]) ** 2, axis=2
        ).array
        correspondence = F.argmin(dists, axis=1).array
        dists = dists[np.arange(dists.shape[0]), correspondence]

        keep = dists < 0.02
        target_match = target[keep]
        correspondence = correspondence[keep]
        source_match = source[correspondence]

        loss = F.sum(
            F.sum((source_match - target_match) ** 2, axis=1), axis=0
        )
        return loss


class ICPRegistration:

    def __init__(
        self,
        points_depth,
        points_cad,
        transform_init=None,
        alpha=0.1,
        gpu=0,
    ):
        quaternion_init = tf.quaternion_from_matrix(transform_init)
        quaternion_init = quaternion_init.astype(np.float32)
        translation_init = tf.translation_from_matrix(transform_init)
        translation_init = translation_init.astype(np.float32)

        link = NearestNeighborICP(quaternion_init, translation_init)

        if gpu >= 0:
            link.to_gpu(gpu)
            points_depth = link.xp.asarray(points_depth)
            points_cad = link.xp.asarray(points_cad)
        self._points_depth = points_depth
        self._points_cad = points_cad

        self._optimizer = chainer.optimizers.Adam(alpha=alpha)
        self._optimizer.setup(link)
        link.translation.update_rule.hyperparam.alpha *= 0.1

    @property
    def _transform(self):
        return cuda.to_cpu(self._optimizer.target.T.array)

    def register_iterative(self, iteration=None):
        iteration = 100 if iteration is None else iteration

        yield self._transform

        for i in range(iteration):
            link = self._optimizer.target
            loss = link(
                source=self._points_cad,
                target=self._points_depth,
            )
            loss.backward()
            self._optimizer.update()
            link.cleargrads()

            # print(f'[{self._iteration:08d}] {loss}')
            # print(f'quaternion:', link.quaternion.array.tolist())
            # print(f'translation:', link.translation.array.tolist())

            yield self._transform


def algorithm():
    gpu = 0
    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()

    instance_id = 3  # == class_id

    models = objslampp.datasets.YCBVideoModels()
    pcd_cad = models.get_pcd(class_id=instance_id)

    dataset = objslampp.datasets.YCBVideoDataset('train')
    example = dataset.get_example(1000)

    depth = example['depth']
    instance_label = example['label']
    K = example['meta']['intrinsic_matrix']
    pcd_depth = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    nonnan = ~np.isnan(pcd_depth).any(axis=2)
    mask = (instance_label == instance_id) & nonnan
    pcd_depth_target = pcd_depth[mask]

    # -------------------------------------------------------------------------

    quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
    translation_init = np.median(pcd_depth_target, axis=0)
    transform_init = objslampp.functions.transformation_matrix(
        quaternion_init, translation_init
    ).array

    pcd_cad = objslampp.extra.open3d.voxel_down_sample(
        pcd_cad, voxel_size=0.01
    )
    pcd_depth_target = objslampp.extra.open3d.voxel_down_sample(
        pcd_depth_target, voxel_size=0.01
    )
    registration = ICPRegistration(pcd_depth_target, pcd_cad, transform_init)

    for T_cad2cam in registration.register_iterative():
        scene = trimesh.Scene()
        geom = trimesh.PointCloud(pcd_depth_target, colors=[1., 0, 0])
        scene.add_geometry(geom, geom_name='a', node_name='a')
        geom = trimesh.PointCloud(pcd_cad, colors=[0, 1., 0])
        scene.add_geometry(
            geom, geom_name='b', node_name='b', transform=T_cad2cam
        )
        scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform()
        yield scene


def main():
    scenes = ({'icp': scene} for scene in algorithm())
    objslampp.extra.trimesh.display_scenes(scenes)


if __name__ == '__main__':
    main()
