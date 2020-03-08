#!/usr/bin/env python

import argparse
import json
import pprint
import re

import chainer
from chainer import cuda
import imgviz
import numpy as np
import path
import pybullet  # NOQA
import trimesh
import trimesh.transformations as tf

import morefusion
from morefusion.contrib import singleview_3d

from train import Transform


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model', help='model file in a log dir')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--save', action='store_true', help='save')
    args = parser.parse_args()

    args_file = path.Path(args.model).parent / 'args'
    with open(args_file) as f:
        args_data = json.load(f)
    pprint.pprint(args_data)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    model = singleview_3d.models.Model(
        n_fg_class=len(args_data['class_names'][1:]),
        pretrained_resnet18=args_data['pretrained_resnet18'],
        with_occupancy=args_data['with_occupancy'],
        loss=args_data['loss'],
        loss_scale=args_data['loss_scale'],
    )
    if args.gpu >= 0:
        model.to_gpu()

    print(f'==> Loading trained model: {args.model}')
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

    split = 'val'
    dataset = morefusion.datasets.YCBVideoRGBDPoseEstimationDataset(
        split=split
    )
    dataset_reindexed = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
        split=split,
        class_ids=args_data['class_ids'],
    )
    transform = Transform(
        train=False,
        with_occupancy=args_data['with_occupancy'],
    )

    pprint.pprint(args.__dict__)

    # -------------------------------------------------------------------------

    depth2rgb = imgviz.Depth2RGB()
    for index in range(len(dataset)):
        frame = dataset.get_frame(index)

        image_id = dataset._ids[index]
        indices = dataset_reindexed.get_indices_from_image_id(image_id)
        examples = dataset_reindexed[indices]
        examples = [transform(example) for example in examples]

        if not examples:
            continue
        inputs = chainer.dataset.concat_examples(examples, device=args.gpu)

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred, confidence_pred = model.predict(
                class_id=inputs['class_id'],
                rgb=inputs['rgb'],
                pcd=inputs['pcd'],
                pitch=inputs.get('pitch'),
                origin=inputs.get('origin'),
                grid_nontarget_empty=inputs.get('grid_nontarget_empty'),
            )

            indices = model.xp.argmax(confidence_pred.array, axis=1)
            quaternion_pred = quaternion_pred[
                model.xp.arange(quaternion_pred.shape[0]), indices
            ]
            translation_pred = translation_pred[
                model.xp.arange(translation_pred.shape[0]), indices
            ]

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

        # TODO(wkentaro)
        observation_new = {}
        for k, v in observation.items():
            if re.match(r'main/add_or_add_s/[0-9]+/.+', k):
                k_new = '/'.join(k.split('/')[:-1])
                observation_new[k_new] = v
        observation = observation_new

        print(f'[{index:08d}] {observation}')

        # ---------------------------------------------------------------------

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

        Ts = dict(true=Ts_true, pred=Ts_pred)

        vizs = []
        depth_viz = depth2rgb(frame['depth'])
        for which in ['true', 'pred']:
            pybullet.connect(pybullet.DIRECT)
            for i, T in enumerate(Ts[which]):
                cad_file = morefusion.datasets.YCBVideoModels()\
                    .get_cad_file(class_id=class_ids[i])
                morefusion.extra.pybullet.add_model(
                    cad_file,
                    position=tf.translation_from_matrix(T),
                    orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
                )
            rgb_rend, depth_rend, segm_rend = \
                morefusion.extra.pybullet.render_camera(
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
                    add = observation[f'main/add_or_add_s/{class_id:04d}']
                    text.append(
                        f'[{which}] [{class_id:04d}]: '
                        f'add/add_s={add * 100:.1f}cm'
                    )
                text = '\n'.join(text)
            else:
                text = f'[{which}]'
            viz = imgviz.draw.text_in_rectangle(
                viz,
                loc='lt',
                text=text,
                size=20,
                background=(0, 255, 0),
                color=(0, 0, 0),
            )
            if which == 'true':
                viz = imgviz.draw.text_in_rectangle(
                    viz,
                    loc='rt',
                    text='singleview_3d',
                    size=20,
                    background=(255, 0, 0),
                    color=(0, 0, 0),
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
