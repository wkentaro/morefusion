#!/usr/bin/env python

import argparse

import numpy as np
import trimesh.transformations as tf
import scipy.io

import objslampp

import contrib
import preliminary


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--icp', action='store_true')
args = parser.parse_args()


dataset = objslampp.datasets.YCBVideoDataset
models = objslampp.datasets.YCBVideoModels()

if args.icp:
    name = 'Densefusion_occupancy_icp_result'
else:
    name = 'Densefusion_occupancy_result'
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
    pcd_scene = objslampp.geometry.pointcloud_from_depth(
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

    # import sys
    # sys.path.insert(0, '../preliminary')
    # from align_occupancy_grids import refinement
    # label_instance = np.zeros(rgb.shape[:2], dtype=np.int32)
    # for ins_id, mask in zip(result['labels'], result['masks']):
    #     mask = mask.astype(bool)
    #     label_instance[mask] = ins_id
    # Ts = np.array([objslampp.geometry.compose_transform(
    #             R=tf.quaternion_matrix(pose[:4])[:3, :3],
    #             t=pose[4:],
    #         ) for pose in result['poses']])
    # refinement(
    #     result['labels'],
    #     result['labels'],
    #     frame['color'],
    #     pcd_scene,
    #     label_instance,
    #     Ts,
    #     Ts,
    # )

    with objslampp.utils.timer(frame_id):
        poses_refined = np.zeros_like(result['poses'])
        for i, (cls_id, mask, pose) in enumerate(zip(
            result['labels'], result['masks'], result['poses']
        )):
            transform_init = objslampp.geometry.compose_transform(
                R=tf.quaternion_matrix(pose[:4])[:3, :3],
                t=pose[4:],
            )
            pcd_cad = np.loadtxt(models.get_pcd_model(class_id=cls_id))

            mask = mask.astype(bool) & nonnan
            pcd_depth = pcd_scene[mask]

            dimensions = np.array([16, 16, 16])
            cad_file = models.get_cad_model(cls_id)
            bbox_diagonal = np.linalg.norm(
                np.nanmax(pcd_cad, axis=0) - np.nanmin(pcd_cad, axis=0)
            )
            pitch = bbox_diagonal / 16
            center = pose[4:]
            origin = center - ((dimensions / 2 - 0.5) * pitch)
            grids = mapping.get_target_grids(
                target_id=cls_id,
                dimensions=dimensions,
                pitch=pitch,
                origin=origin,
            )
            grids = np.stack(grids).astype(np.float32)

            pcd_cad = objslampp.extra.open3d.voxel_down_sample(
                pcd_cad, voxel_size=pitch
            )

            registration = preliminary.OccupancyRegistration(
                pcd_cad.astype(np.float32),
                grids,
                pitch=pitch,
                origin=origin.astype(np.float32),
                threshold=2,
                transform_init=transform_init.astype(np.float32),
                gpu=0,
                alpha=0.01,
            )
            with objslampp.utils.timer('register'):
                transform = registration.register(iteration=100)

            if np.isnan(transform).sum():
                transform = transform_init
            elif args.icp:
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

    mat_file = occupancy_dir / result_file.basename()
    scipy.io.savemat(mat_file, result)
    print(mat_file)
