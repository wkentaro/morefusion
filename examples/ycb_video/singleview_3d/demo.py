#!/usr/bin/env python

import argparse
import json
import pprint
import warnings

import chainer
from chainer import cuda
import imgviz
import numpy as np
import path
import pybullet  # NOQA
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
    parser.add_argument(
        '--dataset',
        choices=[
            'my_synthetic',
            'my_real',
            'ycb_video',
            'ycb_video/train',
            'ycb_video/syn',
        ],
        default='ycb_video',
        help='dataset',
    )
    parser.add_argument(
        '--sampling',
        type=int,
    )
    args = parser.parse_args()

    args_file = path.Path(args.model).parent / 'args'
    with open(args_file) as f:
        args_data = json.load(f)
    pprint.pprint(args_data)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    model = contrib.models.BaselineModel(
        n_fg_class=len(args_data['class_names'][1:]),
        freeze_until=args_data['freeze_until'],
        voxelization=args_data.get('voxelization', 'average'),
        use_occupancy=args_data.get('use_occupancy', False),
    )
    if args.gpu >= 0:
        model.to_gpu()

    print(f'==> Loading trained model: {args.model}')
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

    if args.dataset == 'my_synthetic':
        if args.sampling is not None:
            warnings.warn('--sampling is only used with ycb_video dataset')
        args.root_dir = chainer.dataset.get_dataset_directory(
            # plane type
            'wkentaro/objslampp/ycb_video/synthetic_data/20190408_143724.600111',  # NOQA
            # bin type
            # 'wkentaro/objslampp/ycb_video/synthetic_data/20190402_174648.841996',  # NOQA
        )
        dataset = contrib.datasets.MySyntheticDataset(
            args.root_dir, class_ids=args_data['class_ids']
        )
    elif args.dataset == 'my_real':
        args.root_dir = chainer.dataset.get_dataset_directory(
            'wkentaro/objslampp/ycb_video/real_data/20190614_18'
        )
        dataset = contrib.datasets.MyRealDataset(
            args.root_dir, class_ids=args_data['class_ids']
        )
    elif args.dataset.startswith('ycb_video'):
        split = 'val'
        if '/' in args.dataset:
            _, split = args.dataset.split('/')
        dataset = contrib.datasets.YCBVideoDataset(
            split=split,
            class_ids=args_data['class_ids'],
            sampling=args.sampling,
            return_occupancy_grids=args_data.get('use_occupancy', False),
        )
    else:
        raise ValueError(f'unexpected dataset: {args.dataset}')

    def transform(in_data):
        if args_data.get('use_occupancy', False):
            assert 'grid_target' in in_data
            assert 'grid_nontarget' in in_data
            assert 'grid_empty' in in_data

            grid_nontarget_empty = np.maximum(
                in_data['grid_nontarget'], in_data['grid_empty']
            )
            grid_nontarget_empty = np.float64(grid_nontarget_empty > 0.5)
            grid_nontarget_empty[in_data['grid_target'] > 0.5] = 0
            in_data['grid_nontarget_empty'] = grid_nontarget_empty
            in_data.pop('grid_target')
            in_data.pop('grid_nontarget')
            in_data.pop('grid_empty')
        return in_data

    pprint.pprint(args.__dict__)

    # -------------------------------------------------------------------------

    instances = {}

    depth2rgb = imgviz.Depth2RGB()
    for index in range(len(dataset)):
        with chainer.using_config('debug', True):
            examples = dataset.get_examples(index)
            examples = [
                transform(e) for e in examples
                if e['class_id'] in dataset._class_ids
            ]
        if not examples:
            continue
        inputs = chainer.dataset.concat_examples(examples, device=args.gpu)

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred = model.predict(
                class_id=inputs['class_id'],
                pitch=inputs['pitch'],
                origin=inputs['origin'],
                rgb=inputs['rgb'],
                pcd=inputs['pcd'],
                grid_nontarget_empty=inputs.get('grid_nontarget_empty'),
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

        Ts = dict(true=Ts_true, pred=Ts_pred)

        vizs = []
        depth_viz = depth2rgb(frame['depth'])
        for which in ['true', 'pred']:
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
                    .get_cad_file(class_id=class_ids[i])
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
                [
                    frame['rgb'],
                    depth_viz,
                    rgb_input,
                    segm_rend,
                    rgb_rend,
                    depth_rend,
                ],
                (1, 6),
                border=(255, 255, 255),
            )
            viz = imgviz.resize(viz, width=1800)

            if which == 'pred':
                text = []
                for class_id in np.unique(class_ids):
                    add = observation[f'main/add/{class_id:04d}']
                    add_s = observation[f'main/add_s/{class_id:04d}']
                    text.append(
                        f'[{which}] [{class_id:04d}]: add={add * 100:.1f}cm, '
                        f'add_s={add_s * 100:.1f}cm'
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
        viz = imgviz.tile(vizs, (2, 1), border=(255, 255, 255))

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
