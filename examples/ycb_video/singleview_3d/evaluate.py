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

import objslampp

import contrib


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('model', help='model file in a log dir')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
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

    print('==> Loading trained model: {}'.format(args.model))
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

    dataset = contrib.datasets.YCBVideoDataset(
        'val',
        class_ids=args_data['class_ids'],
        return_occupancy_grids=args_data['use_occupancy'],
    )

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

    # -------------------------------------------------------------------------

    observations = []
    for index in range(len(dataset)):
        examples = dataset.get_example(index)
        if not examples:
            continue
        inputs = chainer.dataset.concat_examples(examples, device=args.gpu)

        keep = inputs['class_id'] != -1
        for key in inputs.keys():
            inputs[key] = inputs[key][keep]
        # make sure there are no multiple instances
        assert np.unique(
            cuda.to_cpu(inputs['class_id']), return_counts=True
        )[1].max() == 1

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
            observations.append(observation)

        print(f'[{index:08d}] {observation}')

    df = pandas.DataFrame(observations)
    out_file = path.Path(args.model).parent / 'add.csv'
    df.to_csv(out_file)
    print('==> Saved ADD CSV:', out_file)

    aucs = {}
    for class_id, class_name in enumerate(args_data['class_names']):
        if class_id == 0:
            continue
        if args_data['class_ids'] and class_id not in args_data['class_ids']:
            continue

        try:
            add = df[f'main/add/{class_id:04d}'].dropna().values
        except KeyError:
            continue
        auc, x, y = objslampp.metrics.ycb_video_add_auc(add, return_xy=True)
        print('AUC(ADD):', auc)

        add_s = df[f'main/add_s/{class_id:04d}'].dropna().values
        auc_s, x_s, y_s = objslampp.metrics.ycb_video_add_auc(
            add_s, return_xy=True)
        print('AUC (ADD-S):', auc_s)

        aucs[class_id] = {'add': auc, 'add_s': auc_s}

        fig = plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.title('ADD (AUC={:.1f})'.format(auc * 100))
        plt.plot(x, y, color='b')
        plt.xlim(0, 0.1)
        plt.ylim(0, 1)
        plt.xlabel('average distance threshold [m]')
        plt.ylabel('accuracy')

        plt.subplot(122)
        plt.title('ADD (AUC={:.1f})'.format(auc_s * 100))
        plt.plot(x_s, y_s, color='b')
        plt.xlim(0, 0.1)
        plt.ylim(0, 1)
        plt.xlabel('average distance threshold [m]')
        plt.ylabel('accuracy')

        plt.tight_layout()
        img = imgviz.io.pyplot_fig2arr(fig)
        out_file = path.Path(args.model).parent / f'auc_{class_id:04d}.png'
        print('==> Saved ADD curve plot:', out_file)
        imgviz.io.imsave(out_file, img)

    df = pandas.DataFrame(aucs)
    out_file = path.Path(args.model).parent / 'auc.csv'
    df.to_csv(out_file)
    print('==> Saved AUC CSV:', out_file)


if __name__ == '__main__':
    main()
