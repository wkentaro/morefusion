#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf

import morefusion

import preliminary


def algorithm():
    models = morefusion.datasets.YCBVideoModels()
    dataset = morefusion.datasets.YCBVideoDataset('train')
    frame = dataset[0]

    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    depth = frame['depth']
    height, width = rgb.shape[:2]
    pcd = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    models = morefusion.datasets.YCBVideoModels()
    class_id = class_ids[0]
    pcd_cad = models.get_pcd(class_id=class_id)

    instance_ids_all = np.r_[0, instance_ids]

    # build octrees
    pitch = 0.005
    mapping = morefusion.contrib.MultiInstanceOctreeMapping()
    for ins_id in instance_ids_all:
        mask = instance_label == ins_id
        mapping.initialize(ins_id, pitch=pitch)
        mapping.integrate(ins_id, mask, pcd)

    target_id = class_id
    mask = instance_label == target_id

    centroid = np.nanmean(pcd[mask], axis=0)
    models = morefusion.datasets.YCBVideoModels()
    diagonal = models.get_bbox_diagonal(class_id=target_id)
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

    # -------------------------------------------------------------------------

    pcd_cad = morefusion.extra.open3d.voxel_down_sample(
        pcd_cad, voxel_size=0.01
    )
    pcd_depth_target = morefusion.extra.open3d.voxel_down_sample(
        occupied_t, voxel_size=0.01
    )
    pcd_depth_nontarget = morefusion.extra.open3d.voxel_down_sample(
        np.vstack((occupied_u, empty)), voxel_size=0.01
    )

    # T_cad2cam
    transform_init = tf.translation_matrix(pcd_depth_target.mean(axis=0))

    registration = preliminary.OccupancyPointsRegistration(
        pcd_depth_target=pcd_depth_target,
        pcd_depth_nontarget=pcd_depth_nontarget,
        pcd_cad=pcd_cad,
        transform_init=transform_init,
    )

    for T_cad2cam in registration.register_iterative():
        camera = trimesh.scene.Camera(
            resolution=(640, 480),
            fov=(60, 45),
        )
        camera_transform = morefusion.extra.trimesh.to_opengl_transform()

        scenes = {}

        scenes['pcd'] = trimesh.Scene(
            camera=camera, camera_transform=camera_transform
        )
        geom = trimesh.PointCloud(pcd_depth_target, colors=[1., 0, 0])
        scenes['pcd'].add_geometry(geom, geom_name='a', node_name='a')
        geom = trimesh.PointCloud(pcd_cad, colors=[0, 1., 0])
        scenes['pcd'].add_geometry(
            geom, geom_name='b', node_name='b', transform=T_cad2cam
        )

        scenes['cad'] = trimesh.Scene(
            camera=camera, camera_transform=camera_transform
        )
        geom = trimesh.PointCloud(pcd_depth_target, colors=[1., 0, 0])
        scenes['cad'].add_geometry(geom, geom_name='a', node_name='a')
        geom = models.get_cad(class_id=target_id)
        scenes['cad'].add_geometry(
            geom, geom_name='b', node_name='b', transform=T_cad2cam
        )

        yield scenes


if __name__ == '__main__':
    morefusion.extra.trimesh.display_scenes(algorithm())
