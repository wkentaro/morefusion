#!/usr/bin/env python

import argparse
import json
import pprint

import chainer
from chainer import cuda
import imgviz
import numpy as np
import path
import pybullet  # NOQA
import pykalman
import trimesh
import trimesh.transformations as tf

import objslampp

import contrib


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model', help='model file in a log dir')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--save', action='store_true', help='save')
    parser.add_argument('--kalman', action='store_true', help='vote')
    args = parser.parse_args()

    args_file = path.Path(args.model).parent / 'args'
    with open(args_file) as f:
        args_data = json.load(f)
    pprint.pprint(args_data)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    model = contrib.models.BaselineModel(
        freeze_until=args_data['freeze_until'],
        voxelization=args_data.get('voxelization', 'average'),
    )
    if args.gpu >= 0:
        model.to_gpu()

    print('==> Loading trained model: {}'.format(args.model))
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

    args.root_dir = chainer.dataset.get_dataset_directory(
        # plane type
        'wkentaro/objslampp/ycb_video/synthetic_data/20190408_143724.600111',
        # bin type
        # 'wkentaro/objslampp/ycb_video/synthetic_data/20190402_174648.841996',
    )
    args.class_ids = [2]
    dataset = contrib.datasets.MySyntheticDataset(
        args.root_dir, class_ids=args.class_ids
    )

    # -------------------------------------------------------------------------

    instances = {}

    depth2rgb = imgviz.Depth2RGB()
    for index in range(len(dataset)):
        with chainer.using_config('debug', True):
            examples = dataset.get_examples(index)
        inputs = chainer.dataset.concat_examples(examples, device=args.gpu)

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred = model.predict(
                class_id=inputs['class_id'],
                pitch=inputs['pitch'],
                rgb=inputs['rgb'],
                pcd=inputs['pcd'],
            )

            reporter = chainer.Reporter()
            reporter.add_observer('main', model)
            observation = {}
            with reporter.scope(observation):
                model.evaluate(
                    class_id=inputs['class_id'],
                    quaternion_true=inputs['quaternion_true'],
                    translation_true=inputs['translation_true'],
                    quaternion_pred=quaternion_pred,
                    translation_pred=translation_pred,
                )

        print(f'[{index:08d}] {observation}')

        # ---------------------------------------------------------------------

        frame = dataset.get_frame(index)
        K = frame['intrinsic_matrix']
        T_cam2world = frame['T_cam2world']
        T_world2cam = np.linalg.inv(T_cam2world)
        height, width = frame['rgb'].shape[:2]
        fovy = trimesh.scene.Camera(
            resolution=(width, height), focal=(K[0, 0], K[1, 1])
        ).fov[1]

        batch_size = len(inputs['class_id'])
        class_ids = cuda.to_cpu(inputs['class_id'])
        quaternion_pred = cuda.to_cpu(quaternion_pred.array)
        translation_pred = cuda.to_cpu(translation_pred.array)
        quaternion_true = cuda.to_cpu(inputs['quaternion_true'])
        translation_true = cuda.to_cpu(inputs['translation_true'])

        def instance_id_from_batch_index(batch_index):
            indices = np.where(np.isin(frame['class_ids'], class_ids))[0]
            index = indices[batch_index]
            assert frame['class_ids'][index] == class_ids[batch_index]
            return frame['instance_ids'][index]

        Ts_pred = []
        Ts_true = []
        for i in range(batch_size):
            # T_cad2cam
            T_pred = tf.quaternion_matrix(quaternion_pred[i])
            T_pred[:3, 3] = translation_pred[i]
            T_true = tf.quaternion_matrix(quaternion_true[i])
            T_true[:3, 3] = translation_true[i]
            Ts_pred.append(T_pred)
            Ts_true.append(T_true)

            instance_id = instance_id_from_batch_index(i)

            if not args.kalman:
                # check if poor prediction
                cad_file = objslampp.datasets.YCBVideoModels()\
                    .get_cad_model(class_id=class_ids[i])
                _, _, mask_rend = \
                    objslampp.extra.pybullet.render_cad(
                        cad_file, Ts_pred[i], fovy, height, width
                    )
                mask_real = frame['instance_label'] == instance_id
                mask_intersect = mask_real & mask_rend
                ratio_cover = mask_intersect.sum() / mask_real.sum()
                is_poor = ratio_cover < 0.85
                if is_poor:
                    continue

            # T_cad2world = T_cam2world @ T_cad2cam
            T_cad2world = T_cam2world @ T_pred
            if instance_id in instances:
                if not instances[instance_id]['spawn']:
                    instances[instance_id]['Ts_cad2world'].append(T_cad2world)
                    instances[instance_id]['batch_index'] = i
            else:
                instances[instance_id] = dict(
                    batch_index=i,
                    class_id=class_ids[i],
                    Ts_cad2world=[T_cad2world],
                    T_cad2world_vote=None,
                    spawn=False,
                )

        # check consistency of pose estimation
        Ts_vote = [None] * len(Ts_pred)
        for instance_id, data in instances.items():
            class_id = data['class_id']
            Ts_cad2world = data['Ts_cad2world']

            if data['spawn']:
                assert data['T_cad2world_vote'] is not None
                T_cad2world = data['T_cad2world_vote']
                T_cad2cam = T_world2cam @ T_cad2world
                Ts_vote[data['batch_index']] = T_cad2cam
                continue

            if len(Ts_cad2world) == 1:
                continue

            if args.kalman:
                Ts_cad2world = np.array(Ts_cad2world)[:, :3, :]
                kf = pykalman.KalmanFilter(
                    transition_matrices=np.eye(12),
                    observation_matrices=np.eye(12),
                    n_dim_obs=12,
                    n_dim_state=12,
                )
                kf.em(Ts_cad2world.reshape(-1, 12))
                means, covariances = kf.smooth(Ts_cad2world.reshape(-1, 12))
                stds = np.sqrt(covariances[:, np.arange(12), np.arange(12)])

                means = means.reshape(-1, 3, 4)
                stds = stds.reshape(-1, 3, 4)

                std_rotations = stds[:, :3, :3]
                std_rotations[:, np.arange(3), np.arange(3)] += 1
                std_translations = stds[:, :3, 3]

                std_quaternions = np.array([
                    tf.quaternion_from_matrix(r) for r in std_rotations
                ])
                qws = std_quaternions[:, 0]
                angles = np.arccos(qws) * 2
                distances = np.linalg.norm(std_translations, axis=1)

                index = np.argmin(angles)
                print(f'uncertainty: {np.rad2deg(angles[index])} [deg], '
                      f'{distances[index] * 100} [cm]')
                if angles[index] < np.deg2rad(0.3):
                    if distances[index] < 0.03:
                        data['spawn'] = True
                        data['T_cad2world_vote'] = np.r_[
                            means[index], [[0, 0, 0, 1]]
                        ]
                        T_cad2cam = T_world2cam @ data['T_cad2world_vote']
                        assert Ts_vote[data['batch_index']] is None
                        Ts_vote[data['batch_index']] = T_cad2cam
            else:
                pcd_file = objslampp.datasets.YCBVideoModels()\
                    .get_model(class_id=class_id)['points_xyz']
                pcd = np.loadtxt(pcd_file)
                Ts_prev = Ts_cad2world[:-1]
                adds = objslampp.metrics.average_distance(
                    [pcd] * len(Ts_prev),
                    [Ts_cad2world[-1]] * len(Ts_prev),
                    Ts_prev,
                )
                # there are at least N hypothesis around Xcm
                if (adds < 0.02).sum() >= 3:
                    data['spawn'] = True
                    data['T_cad2world_vote'] = Ts_cad2world[-1]
                    T_cad2cam = T_world2cam @ data['T_cad2world_vote']
                    assert Ts_vote[data['batch_index']] is None
                    Ts_vote[data['batch_index']] = T_cad2cam

        Ts = dict(true=Ts_true, pred=Ts_pred, vote=Ts_vote)

        vizs = []
        for which in ['true', 'pred', 'vote']:
            pybullet.connect(pybullet.DIRECT)
            for i, T in enumerate(Ts[which]):
                instance_id = instance_id_from_batch_index(i)
                if instance_id in instances:
                    assert instances[instance_id]['batch_index'] == i
                if (which == 'vote' and
                        (instance_id not in instances or
                            not instances[instance_id]['spawn'])):
                    continue
                cad_file = objslampp.datasets.YCBVideoModels()\
                    .get_model(class_id=class_ids[i])['textured_simple']
                objslampp.extra.pybullet.add_model(
                    cad_file,
                    position=tf.translation_from_matrix(T),
                    orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
                )
            rgb_rend, depth_rend, segm_rend = \
                objslampp.extra.pybullet.render_camera(
                    np.eye(4), fovy, height, width
                )
            pybullet.disconnect()

            segm_rend = imgviz.label2rgb(
                segm_rend + 1, img=frame['rgb'], alpha=0.7
            )
            depth_rend = depth2rgb(depth_rend)
            rgb_input = imgviz.tile(
                cuda.to_cpu(inputs['rgb']), border=(255, 255, 255)
            )
            viz = imgviz.tile(
                [frame['rgb'], rgb_input, segm_rend, rgb_rend, depth_rend],
                (1, 5),
                border=(255, 255, 255),
            )
            viz = imgviz.resize(viz, width=1800)

            if which == 'pred':
                text = []
                for class_id in np.unique(class_ids):
                    add = observation[f'main/add/{class_id:04d}']
                    add_rot = observation[f'main/add_rotation/{class_id:04d}']
                    text.append(
                        f'[{which}] [{class_id:04d}]: add={add * 100:.1f}cm, '
                        f'add_rot={add_rot * 100:.1f}cm'
                    )
                text = '\n'.join(text)
            else:
                text = f'[{which}]'
            viz = imgviz.draw.text_in_rectangle(
                viz, loc='lt', text=text, size=20, background=(0, 255, 0)
            )
            if which == 'true':
                viz = imgviz.draw.text_in_rectangle(
                    viz,
                    loc='rt',
                    text='singleview_3d',
                    size=20,
                    background=(255, 0, 0),
                )
            vizs.append(viz)
        viz = imgviz.tile(vizs, (3, 1), border=(255, 255, 255))

        if args.save:
            out_file = path.Path(args.model).parent / f'video/{index:08d}.jpg'
            out_file.parent.makedirs_p()
            imgviz.io.imsave(out_file, viz)

        yield viz


if __name__ == '__main__':
    vizs = main()
    try:
        imgviz.io.pyglet_imshow(vizs)
        imgviz.io.pyglet_run()
    except StopIteration:
        pass
