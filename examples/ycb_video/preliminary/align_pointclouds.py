#!/usr/bin/env python

import argparse

import imgviz
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tf

import objslampp

import contrib


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--case', type=int, default=0, choices=[0, 1, 2], help='case'
    )
    args = parser.parse_args()

    if args.case == 0:
        class_id = 2
        R = tf.rotation_matrix(np.deg2rad(180), [1, 0, 0])
        quaternion_init = tf.quaternion_from_matrix(R).astype(np.float32)
        translation_init = np.zeros((3,), dtype=np.float32)
    elif args.case == 1:
        class_id = 20
        np.random.seed(4)
        quaternion_init = tf.random_quaternion().astype(np.float32)
        translation_init = np.random.uniform(
            -0.1, 0.1, (3,)
        ).astype(np.float32)
    elif args.case == 2:
        class_id = 20
        R = tf.rotation_matrix(np.deg2rad(90), [1, 0, 0])
        R = tf.rotation_matrix(np.deg2rad(180), [0, 1, 0]) @ R
        quaternion_init = tf.quaternion_from_matrix(R).astype(np.float32)
        translation_init = np.zeros((3,), dtype=np.float32)

    models = objslampp.datasets.YCBVideoModels()
    pcd_file = models.get_pcd_model(class_id=class_id)
    points = np.loadtxt(pcd_file, dtype=np.float32)

    indices = np.random.permutation(len(points))[:1000]
    points_target = points[indices]
    indices = np.random.permutation(len(points))[:500]
    points_source = points[indices]

    T = tf.quaternion_matrix(quaternion_init)
    T = objslampp.geometry.compose_transform(T[:3, :3], translation_init)
    points_source = tf.transform_points(points_source, T)

    transform_pred = contrib.icp.register_pointcloud(
        points_source, points_target, debug=True
    )

    for index, transform_pred in enumerate([np.eye(4), transform_pred]):
        transform_pred = transform_pred @ T

        scene = pyrender.Scene()

        cad_file = models.get_cad_model(class_id=class_id)
        cad = trimesh.load(str(cad_file))
        cad.visual = cad.visual.to_color()

        cad_true = cad.copy()
        cad_true.visual.vertex_colors[:, 3] = 200
        scene.add(pyrender.Mesh.from_trimesh(cad_true, smooth=False))

        cad_pred = cad.copy()
        scene.add(
            pyrender.Mesh.from_trimesh(cad_pred, smooth=False),
            pose=transform_pred,
        )

        eye = objslampp.geometry.points_from_angles(
            distance=0.4, elevation=30, azimuth=45
        )
        cam_pose = objslampp.geometry.look_at(eye)
        cam_pose = objslampp.extra.trimesh.camera_transform(cam_pose)

        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45))
        scene.add(camera, pose=cam_pose)

        spotlight = pyrender.SpotLight(intensity=3)
        scene.add(spotlight, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(640, 480)
        rgb = renderer.render(scene)[0][:, :, :3]
        imgviz.io.imsave(f'logs/icp_{index:04d}.jpg', rgb)


if __name__ == '__main__':
    main()
