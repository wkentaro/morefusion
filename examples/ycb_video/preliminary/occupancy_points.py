#!/usr/bin/env python

import chainer
import chainer.functions as F
import numpy as np
import trimesh

import objslampp

import preliminary


class OccupancyPointsRegistrationLink(chainer.Link):

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

    def forward(
        self,
        pcd_depth_target,
        pcd_depth_nontarget,
        pcd_cad,
        threshold_nontarget=0.1,
    ):
        # source is transformed
        # source is the starting point for nearest neighbor

        loss = 0
        for i in [0, 1]:
            if i == 0:
                source = pcd_depth_target
                target = pcd_cad
            else:
                source = pcd_depth_nontarget
                target = pcd_cad

            source = objslampp.functions.transform_points(
                source, self.T[None])[0]

            dists = F.sum(
                (source[None, :, :] - target[:, None, :]) ** 2, axis=2
            ).array
            correspondence = F.argmin(dists, axis=0).array
            dists = dists[correspondence, np.arange(dists.shape[1])]

            if i == 0:
                keep = dists < 0.02
                source_match = source[keep]
                correspondence = correspondence[keep]
                target_match = target[correspondence]

                dists_match = F.sum((source_match - target_match) ** 2, axis=1)
                loss_i = F.mean(dists_match, axis=0) / 0.02
                loss += loss_i
            elif threshold_nontarget > 0:
                keep = dists < threshold_nontarget
                source_match = source[keep]
                correspondence = correspondence[keep]
                target_match = target[correspondence]

                dists_match = F.sum((source_match - target_match) ** 2, axis=1)
                loss_i = F.mean(0.1 - dists_match) / 0.1
                loss += loss_i
        return loss


def algorithm():
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset[0]

    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    depth = frame['depth']
    height, width = rgb.shape[:2]
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    models = objslampp.datasets.YCBVideoModels()
    class_id = class_ids[0]
    pcd_file = models.get_pcd_model(class_id=class_id)
    pcd_cad = np.loadtxt(pcd_file)

    instance_ids_all = np.r_[0, instance_ids]

    # build octrees
    pitch = 0.005
    mapping = preliminary.MultiInstanceOctreeMapping()
    for ins_id in instance_ids_all:
        mask = instance_label == ins_id
        mapping.initialize(ins_id, pitch=pitch)
        mapping.integrate(ins_id, mask, pcd)

    target_id = 8
    mask = instance_label == target_id

    centroid = np.nanmean(pcd[mask], axis=0)
    models = objslampp.datasets.YCBVideoModels()
    diagonal = models.get_bbox_diagonal(models.get_cad_model(target_id))
    aabb_min = centroid - diagonal / 2
    aabb_max = aabb_min + diagonal
    print(aabb_min, aabb_max)

    occupied_t, empty_i = mapping.get_target_pcds(
        target_id, aabb_min, aabb_max
    )
    occupied_u = []
    empty = [empty_i]
    for ins_id in instance_ids_all:
        if ins_id == target_id:
            continue
        occupied_u_i, empty_i = mapping.get_target_pcds(
            ins_id, aabb_min, aabb_max
        )
        occupied_u.append(occupied_u_i)
        empty.append(empty_i)
    occupied_u = np.concatenate(occupied_u, axis=0)
    empty = np.concatenate(empty, axis=0)

    print(occupied_t.shape)
    print(occupied_u.shape)
    print(empty.shape)

    # camera = trimesh.scene.Camera(
    #     resolution=(640, 480),
    #     focal=(K[0, 0], K[1, 1]),
    #     transform=objslampp.extra.trimesh.to_opengl_transform(),
    # )
    # scene = trimesh.Scene(camera=camera)
    # geom = trimesh.PointCloud(vertices=occupied_t, colors=(1., 0, 0))
    # scene.add_geometry(geom)
    # geom = trimesh.PointCloud(vertices=occupied_u, colors=(0, 1., 0))
    # scene.add_geometry(geom)
    # geom = trimesh.PointCloud(vertices=empty, colors=(0.5, 0.5, 0.5, 0.5))
    # scene.add_geometry(geom)
    # geom = trimesh.path.creation.box_outline((aabb_max - aabb_min))
    # geom.apply_translation(centroid)
    # scene.add_geometry(geom)
    # preliminary.display_scenes({__file__: scene})

    # -------------------------------------------------------------------------

    pcd_cad = objslampp.extra.open3d.voxel_down_sample(
        pcd_cad, voxel_size=0.01
    )
    pcd_depth_target = objslampp.extra.open3d.voxel_down_sample(
        occupied_t, voxel_size=0.01
    )
    pcd_depth_nontarget = objslampp.extra.open3d.voxel_down_sample(
        np.vstack((occupied_u, empty)), voxel_size=0.01
    )

    quaternion_init = np.array([1, 0, 0, 0], dtype=np.float32)
    translation_init = - pcd_depth_target.mean(axis=0)
    # from icp import NearestNeighborICP
    # link = NearestNeighborICP(
    #     quaternion_init=quaternion_init,
    #     translation_init=translation_init,
    # )
    link = OccupancyPointsRegistrationLink(
        quaternion_init=quaternion_init,
        translation_init=translation_init,
    )

    optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer.setup(link)
    link.translation.update_rule.hyperparam.alpha *= 0.1

    for i in range(300):
        # T_cam2cad = cuda.to_cpu(link.T.array)
        T_cam2cad = link.T.array
        T_cad2cam = np.linalg.inv(T_cam2cad)

        scene = trimesh.Scene()
        geom = trimesh.PointCloud(pcd_depth_target, colors=[1., 0, 0])
        scene.add_geometry(geom, geom_name='a', node_name='a')
        geom = trimesh.PointCloud(pcd_cad, colors=[0, 1., 0])
        scene.add_geometry(
            geom, geom_name='b', node_name='b', transform=T_cad2cam
        )
        scene.camera.transform = objslampp.extra.trimesh.camera_transform()
        yield scene

        optimizer.target.cleargrads()
        loss = optimizer.target(
            pcd_depth_target=pcd_depth_target,
            pcd_depth_nontarget=pcd_depth_nontarget,
            pcd_cad=pcd_cad,
            threshold_nontarget=0.1 / (i + 1),
        )
        loss.backward()
        optimizer.update()


def main():
    import preliminary

    scenes = ({'icp': scene} for scene in algorithm())
    preliminary.display_scenes(scenes)


if __name__ == '__main__':
    main()
