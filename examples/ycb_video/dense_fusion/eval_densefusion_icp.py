#!/usr/bin/env python

import numpy as np
import trimesh.transformations as tf
import scipy.io

import objslampp

import contrib
import preliminary


dataset = objslampp.datasets.YCBVideoDataset
models = objslampp.datasets.YCBVideoModels()

icp_dir = contrib.get_eval_result(name='Densefusion_icp_result')
icp_dir.mkdir_p()

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
    pcd_scene = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
    )

    with objslampp.utils.timer(frame_id):
        poses_refined = np.zeros_like(result['poses'])
        for i, (cls_id, mask, pose) in enumerate(zip(
            result['labels'], result['masks'], result['poses']
        )):
            transform_init = objslampp.geometry.compose_transform(
                R=tf.quaternion_matrix(pose[:4])[:3, :3],
                t=pose[4:],
            )
            pcd_cad = np.loadtxt(models.get_pcd_file(class_id=cls_id))

            mask = mask.astype(bool) & nonnan
            pcd_depth = pcd_scene[mask]

            registration = preliminary.ICPRegistration(
                pcd_depth, pcd_cad, transform_init
            )
            transform = registration.register()

            pose_refined = np.r_[
                tf.quaternion_from_matrix(transform),
                tf.translation_from_matrix(transform),
            ]
            poses_refined[i] = pose_refined

    result['poses'] = poses_refined

    mat_file = icp_dir / result_file.basename()
    scipy.io.savemat(mat_file, result)
    print(mat_file)
