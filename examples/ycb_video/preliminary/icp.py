#!/usr/bin/env python

import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np
import trimesh

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
        # source -> target
        R = objslampp.functions.quaternion_matrix(self.quaternion[None])[0]
        T = objslampp.functions.translation_matrix(self.translation[None])[0]
        return R @ T

    def forward(self, source, target):
        # source is transformed
        # source is the starting point for nearest neighbor

        source = objslampp.functions.transform_points(source, self.T[None])[0]

        dists = F.sum(
            (source[None, :, :] - target[:, None, :]) ** 2, axis=2
        ).array
        correspondence = F.argmin(dists, axis=0).array
        dists = dists[correspondence, np.arange(dists.shape[1])]

        keep = dists < 0.02
        source_match = source[keep]
        correspondence = correspondence[keep]
        target_match = target[correspondence]

        loss = F.sum(
            F.sum((source_match - target_match) ** 2, axis=1), axis=0
        )
        return loss


def algorithm():
    gpu = 0
    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()

    models = objslampp.datasets.YCBVideoModels()
    pcd_file = models.get_pcd_file(class_id=2)
    pcd_cad = np.loadtxt(pcd_file)

    dataset = objslampp.datasets.YCBVideoDataset('train')
    example = dataset.get_example(1000)

    depth = example['depth']
    instance_label = example['label']
    instance_id = 2
    K = example['meta']['intrinsic_matrix']
    pcd_depth = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    nonnan = ~np.isnan(pcd_depth).any(axis=2)
    mask = (instance_label == instance_id) & nonnan
    pcd_depth_target = pcd_depth[mask]

    # -------------------------------------------------------------------------

    pcd_cad = objslampp.extra.open3d.voxel_down_sample(
        pcd_cad, voxel_size=0.01
    )
    pcd_depth_target = objslampp.extra.open3d.voxel_down_sample(
        pcd_depth_target, voxel_size=0.01
    )

    pcd_cad = pcd_cad.astype(np.float32)
    pcd_depth_target = pcd_depth_target.astype(np.float32)
    pcd_cad_cpu = pcd_cad
    pcd_depth_target_cpu = pcd_depth_target
    if gpu >= 0:
        pcd_cad = cuda.to_gpu(pcd_cad)
        pcd_depth_target = cuda.to_gpu(pcd_depth_target)

    quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
    translation_init = - pcd_depth_target.mean(axis=0)
    nnicp = NearestNeighborICP(
        quaternion_init=quaternion_init,
        translation_init=translation_init,
    )
    if gpu >= 0:
        nnicp.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer.setup(nnicp)
    nnicp.translation.update_rule.hyperparam.alpha *= 0.1

    for i in range(300):
        T_cam2cad = cuda.to_cpu(nnicp.T.array)
        T_cad2cam = np.linalg.inv(T_cam2cad)

        scene = trimesh.Scene()
        geom = trimesh.PointCloud(pcd_depth_target_cpu, colors=[1., 0, 0])
        scene.add_geometry(geom, geom_name='a', node_name='a')
        geom = trimesh.PointCloud(pcd_cad_cpu, colors=[0, 1., 0])
        scene.add_geometry(
            geom, geom_name='b', node_name='b', transform=T_cad2cam
        )
        scene.camera.transform = objslampp.extra.trimesh.camera_transform()
        yield scene

        optimizer.target.cleargrads()
        loss = optimizer.target(source=pcd_depth_target, target=pcd_cad)
        loss.backward()
        optimizer.update()


def main():
    import preliminary

    scenes = ({'icp': scene} for scene in algorithm())
    objslampp.extra.trimesh.display_scenes(scenes)


if __name__ == '__main__':
    main()
