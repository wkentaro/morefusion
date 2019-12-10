#!/usr/bin/env python

import argparse
import datetime
import shutil

import chainer
import numpy as np
import path
import pybullet

import morefusion


def generate_a_video(out, random_state, connection_method=None):
    out.makedirs_p()
    (out / 'models').mkdir_p()

    models = morefusion.datasets.YCBVideoModels()

    class_weight = np.zeros((models.n_class - 1,), dtype=float)
    class_weight[...] = 1
    class_weight /= class_weight.sum()

    generator = morefusion.simulation.BinTypeSceneGeneration(
        extents=random_state.uniform((0.2, 0.2, 0.2), (0.5, 0.5, 0.3)),
        models=models,
        n_object=random_state.randint(8, 14),
        random_state=random_state,
        class_weight=class_weight,
        multi_instance=True,
        connection_method=connection_method,
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-60,
        cameraTargetPosition=(0, 0, 0),
    )
    generator.generate()

    cad_files = {}
    for ins_id, data in generator._objects.items():
        if 'cad_file' in data:
            dst_file = out / f'models/{ins_id:08d}.obj'
            print(f'Saved: {dst_file}')
            shutil.copy(data['cad_file'], dst_file)
            cad_files[ins_id] = f'models/{ins_id:08d}.obj'

    Ts_cam2world = generator.random_camera_trajectory(
        n_keypoints=5, n_points=15,
    )
    camera = morefusion.extra.trimesh.OpenGLCamera(
        resolution=(640, 480), fovy=45
    )

    for index, T_cam2world in enumerate(Ts_cam2world):
        rgb, depth, instance_label, class_label = generator.render(
            T_cam2world,
            fovy=camera.fov[1],
            height=camera.resolution[1],
            width=camera.resolution[0],
        )
        instance_ids = generator.unique_ids
        cad_ids = generator.unique_ids_to_cad_ids(instance_ids)
        class_ids = generator.unique_ids_to_class_ids(instance_ids)
        Ts_cad2world = generator.unique_ids_to_poses(instance_ids)
        T_world2cam = np.linalg.inv(T_cam2world)
        Ts_cad2cam = T_world2cam @ Ts_cad2world

        # validation
        n_instance = len(instance_ids)
        assert len(Ts_cad2cam) == n_instance
        assert len(class_ids) == n_instance

        width, height = camera.resolution
        assert rgb.shape == (height, width, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (height, width)
        assert depth.dtype == np.float32
        assert instance_label.shape == (height, width)
        assert instance_label.dtype == np.int32
        assert class_label.shape == (height, width)
        assert class_label.dtype == np.int32

        assert Ts_cad2cam.shape == (n_instance, 4, 4)
        assert Ts_cad2cam.dtype == np.float64
        assert T_cam2world.shape == (4, 4)
        assert T_cam2world.dtype == np.float64

        data = dict(
            rgb=rgb,
            depth=depth,
            instance_label=instance_label,
            class_label=class_label,
            intrinsic_matrix=camera.K,
            T_cam2world=T_cam2world,
            Ts_cad2cam=Ts_cad2cam,
            instance_ids=instance_ids,
            class_ids=class_ids,
            cad_ids=cad_ids,
            cad_files=[cad_files.get(i, '') for i in instance_ids],
        )

        npz_file = out / f'{index:08d}.npz'
        print(f'==> Saved: {npz_file}')
        np.savez_compressed(npz_file, **data)

    morefusion.extra.pybullet.del_world()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nogui', action='store_true', help='no gui')
    args = parser.parse_args()

    if args.nogui:
        connection_method = pybullet.DIRECT
    else:
        connection_method = pybullet.GUI

    now = datetime.datetime.utcnow()
    timestamp = now.strftime('%Y%m%d_%H%M%S.%f')
    root_dir = chainer.dataset.get_dataset_directory(
        f'wkentaro/morefusion/ycb_video/synthetic_data/{timestamp}'
    )
    root_dir = path.Path(root_dir)

    n_video = 1200
    for index in range(1, n_video + 1):
        video_dir = root_dir / f'{index:08d}'
        random_state = np.random.RandomState(index)
        generate_a_video(video_dir, random_state, connection_method)


if __name__ == '__main__':
    main()
