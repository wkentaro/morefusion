#!/usr/bin/env python

import numpy as np
import trimesh.transformations as tf
import scipy.io

import morefusion

import contrib
import preliminary


dataset = morefusion.datasets.YCBVideoDataset
models = morefusion.datasets.YCBVideoModels()

name = 'Densefusion_occupancy_points_result'
occupancy_dir = contrib.get_eval_result(name)
occupancy_dir.mkdir_p()

norefine_dir = contrib.get_eval_result(name='Densefusion_wo_refine_result')
for result_file in sorted(norefine_dir.glob('*.mat')):
    result = scipy.io.loadmat(
        result_file, chars_as_strings=True, squeeze_me=True
    )
    frame_id = '/'.join(result['frame_id'].split('/')[1:])

    frame = dataset.get_frame(frame_id)

    rgb = frame['color']
    depth = frame['depth']
    nonnan = ~np.isnan(depth)
    K = frame['meta']['intrinsic_matrix']
    pcd_scene = morefusion.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
    )

    mapping = preliminary.MultiInstanceOctreeMapping()
    mask_bg = np.ones(rgb.shape[:2], dtype=bool)
    for ins_id, mask in zip(result['labels'], result['masks']):
        mask = mask.astype(bool)
        mapping.initialize(ins_id, pitch=0.01)
        mapping.integrate(ins_id, mask, pcd_scene)
        mask_bg = mask_bg & (~mask)
    mapping.initialize(0, pitch=0.01)
    mapping.integrate(0, mask_bg, pcd_scene)

    instance_ids_all = np.r_[0, frame['meta']['cls_indexes']]
    with morefusion.utils.timer(frame_id):
        poses_refined = np.zeros_like(result['poses'])
        for i, (cls_id, mask, pose) in enumerate(zip(
            result['labels'], result['masks'], result['poses']
        )):
            transform_init = morefusion.geometry.compose_transform(
                R=tf.quaternion_matrix(pose[:4])[:3, :3],
                t=pose[4:],
            )
            pcd_cad = models.get_pcd(class_id=cls_id)

            mask = mask.astype(bool) & nonnan
            pcd_depth = pcd_scene[mask]

            centroid = np.mean(pcd_depth, axis=0)
            diagonal = models.get_bbox_diagonal(class_id=cls_id)
            aabb_min = centroid - diagonal / 2
            aabb_max = aabb_min + diagonal

            occupied_t, empty_i = mapping.get_target_pcds(
                cls_id, aabb_min, aabb_max
            )
            occupied_u = []
            empty = [empty_i]
            for ins_id in instance_ids_all:
                if ins_id == cls_id:
                    continue
                occupied_u_i, empty_i = mapping.get_target_pcds(
                    ins_id, aabb_min, aabb_max
                )
                occupied_u.append(occupied_u_i)
                empty.append(empty_i)
            occupied_u = np.concatenate(occupied_u, axis=0)
            empty = np.concatenate(empty, axis=0)

            pcd_cad = morefusion.extra.open3d.voxel_down_sample(
                pcd_cad, voxel_size=0.01
            )
            pcd_depth_target = morefusion.extra.open3d.voxel_down_sample(
                occupied_t, voxel_size=0.01
            )
            pcd_depth_nontarget = morefusion.extra.open3d.voxel_down_sample(
                np.vstack((occupied_u, empty)), voxel_size=0.01
            )

            registration = preliminary.OccupancyPointsRegistration(
                pcd_depth_target.astype(np.float32),
                pcd_depth_nontarget.astype(np.float32),
                pcd_cad.astype(np.float32),
                transform_init=transform_init.astype(np.float32),
                gpu=0,
                alpha=0.01,
            )
            with morefusion.utils.timer('register'):
                transform = registration.register(iteration=100)

            if np.isnan(transform).sum():
                transform = transform_init

            pose_refined = np.r_[
                tf.quaternion_from_matrix(transform),
                tf.translation_from_matrix(transform),
            ]
            poses_refined[i] = pose_refined

    result['poses'] = poses_refined

    mat_file = occupancy_dir / result_file.basename()
    scipy.io.savemat(mat_file, result)
    print(mat_file)
