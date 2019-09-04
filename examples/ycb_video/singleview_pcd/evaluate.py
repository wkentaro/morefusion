#!/usr/bin/env python

import argparse
import json
import pprint

import chainer
from chainer.backends import cuda
import imgviz
import matplotlib.pyplot as plt
import numpy as np
import pandas
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
    parser.add_argument('--show', action='store_true', help='show')
    parser.add_argument('--save', action='store_true', help='save')
    parser.add_argument(
        '--dataset',
        default='ycb_video',
        choices=['ycb_video', 'bin_type'],
        help='dataset',
    )
    parser.add_argument('--instance-id', type=int, help='instance id')
    args = parser.parse_args()

    args_file = path.Path(args.model).parent / 'args'
    with open(args_file) as f:
        args_data = json.load(f)
    pprint.pprint(args_data)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    model = contrib.models.PoseNet(
        n_fg_class=21,
        freeze_until=args_data['freeze_until'],
        lambda_confidence=args_data['lambda_confidence'],
    )
    if args.gpu >= 0:
        model.to_gpu()

    print('==> Loading trained model: {}'.format(args.model))
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

    if args.dataset == 'ycb_video':
        dataset = objslampp.datasets.YCBVideoRGBDPoseEstimationDataset(
            'val', class_ids=[2]
        )
    else:
        assert args.dataset == 'bin_type'
        dataset = contrib.datasets.MySyntheticDataset(
            '/home/wkentaro/data/datasets/wkentaro/objslampp/ycb_video/synthetic_data/20190402_174648.841996',  # NOQA
            class_ids=[2],
        )
        dataset.instance_id = args.instance_id

    # -------------------------------------------------------------------------

    observations = []
    depth2rgb = imgviz.Depth2RGB()
    for index in range(len(dataset)):
        with chainer.using_config('debug', True):
            examples = dataset[index:index + 1]
        inputs = chainer.dataset.concat_examples(examples, device=args.gpu)

        if inputs['class_id'][0] == -1:
            continue

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred, confidence_pred = model.predict(
                class_id=inputs['class_id'],
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
                    confidence_pred=confidence_pred,
                )
            observations.append(observation)

        print(f'[{index:08d}] {observation}')

        if not (args.show or args.save):
            continue

        # ---------------------------------------------------------------------

        class_id = int(inputs['class_id'][0])

        frame = dataset.get_frame(index)
        rgb = frame['rgb']
        K = frame['intrinsic_matrix']
        height, width = rgb.shape[:2]
        fovy = trimesh.scene.Camera(
            resolution=(width, height), focal=(K[0, 0], K[1, 1])
        ).fov[1]
        cad_file = objslampp.datasets.YCBVideoModels()\
            .get_cad_file(class_id=class_id)

        keep = confidence_pred.array.argmax(axis=1)
        quaternion_pred = quaternion_pred[np.arange(1), keep, :]
        translation_pred = translation_pred[np.arange(1), keep, :]

        quaternion_pred = cuda.to_cpu(quaternion_pred.array)[0]
        translation_pred = cuda.to_cpu(translation_pred.array)[0]
        T_pred = tf.quaternion_matrix(quaternion_pred)
        T_pred[:3, 3] = translation_pred
        quaternion_true = cuda.to_cpu(inputs['quaternion_true'])[0]
        translation_true = cuda.to_cpu(inputs['translation_true'])[0]
        T_true = tf.quaternion_matrix(quaternion_true)
        T_true[:3, 3] = translation_true

        Ts = {'true': T_true, 'pred': T_pred}

        vizs = []
        for which in ['true', 'pred']:
            rgb_rend, depth_rend, mask_rend = \
                objslampp.extra.pybullet.render_cad(
                    cad_file, Ts[which], fovy=fovy, height=height, width=width
                )
            mask_rend = imgviz.label2rgb(
                mask_rend.astype(np.int32), img=rgb, alpha=0.7
            )
            depth_rend = depth2rgb(depth_rend)
            viz = imgviz.tile(
                [rgb, examples[0]['rgb'], mask_rend, rgb_rend, depth_rend],
                (1, 5),
                border=(255, 255, 255),
            )
            viz = imgviz.resize(viz, width=1800)

            add = observation[f'main/add/{class_id:04d}']
            add_rot = observation[f'main/add_rotation/{class_id:04d}']
            text = f'[{which}]: add={add * 100:.1f}cm, add_rot={add_rot * 100:.1f}cm' # NOQA
            viz = imgviz.draw.text_in_rectangle(
                viz, loc='lt', text=text, size=20, background=(0, 255, 0),
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

        if args.show:
            yield viz

    df = pandas.DataFrame(observations)

    errors = df['main/add_rotation/0002'].values
    auc, x, y = objslampp.metrics.auc_for_errors(errors, 0.1, return_xy=True)
    print('auc (add_rotation):', auc)

    fig = plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.title('ADD (rotation) (AUC={:.1f})'.format(auc * 100))
    plt.plot(x, y, color='b')
    plt.xlim(0, 0.1)
    plt.ylim(0, 1)
    plt.xlabel('average distance threshold [m]')
    plt.ylabel('accuracy')

    plt.subplot(122)
    errors = df['main/add/0002'].values
    auc, x, y = objslampp.metrics.auc_for_errors(errors, 0.1, return_xy=True)
    print('auc (add):', auc)
    plt.title('ADD (rotation + translation) (AUC={:.1f})'.format(auc * 100))
    plt.plot(x, y, color='b')
    plt.xlim(0, 0.1)
    plt.ylim(0, 1)
    plt.xlabel('average distance threshold [m]')
    plt.ylabel('accuracy')

    plt.tight_layout()
    img = imgviz.io.pyplot_fig2arr(fig)
    out_file = path.Path(args.model).parent / 'auc.png'
    print('==> Saved ADD curve plot:', out_file)
    imgviz.io.imsave(out_file, img)


if __name__ == '__main__':
    vizs = main()
    try:
        imgviz.io.pyglet_imshow(vizs)
        imgviz.io.pyglet_run()
    except StopIteration:
        pass
