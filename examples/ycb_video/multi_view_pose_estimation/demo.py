#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
import imgviz

import objslampp
from objslampp import logger

from view_dataset import MainApp


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--model', help='model file in a log dir')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

# -----------------------------------------------------------------------------

args_file = pathlib.Path(args.model).parent / 'args'
with open(args_file) as f:
    args_data = json.load(f)
pprint.pprint(args_data)

# -----------------------------------------------------------------------------

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()

model = objslampp.models.SimpleMV3DCNNModel(
    extractor=args_data['extractor'],
    lambda_translation=args_data['lambda_translation'],
    lambda_quaternion=args_data['lambda_quaternion'],
)
if args.gpu >= 0:
    model.to_gpu()

# args.model = 'logs.hoop/20190312_235147.939651/snapshot_model_iter_3190.npz'
if args.model is not None:
    print('==> Loading trained model: {}'.format(args.model))
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

dataset = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
    'val',
    class_ids=args_data['class_ids'],
    sampling=60,
    num_frames_scan=args_data.get('num_frames_scan'),
)

# -----------------------------------------------------------------------------

for index in range(len(dataset)):
    examples = dataset[index:index + 1]
    inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
    inputs.pop('valid')
    inputs.pop('gt_pose')
    video_id = int(inputs.pop('video_id')[0])
    gt_quaternion = inputs.pop('gt_quaternion')
    gt_translation = inputs.pop('gt_translation')
    with chainer.no_backprop_mode() and chainer.using_config('train', False):
        quaternion, translation = model.predict(**inputs)
        with chainer.using_config('debug', False):
            loss = model.loss(
                quaternion=quaternion,
                translation=translation,
                gt_quaternion=gt_quaternion,
                gt_translation=gt_translation,
            )
            loss = float(loss.array)
            print(f'[{video_id:04d}] [{index:08d}] {loss}')

            # visualize
            '''
            example = examples[0]
            masks = example['scan_masks']
            rgbs = example['scan_rgbs']
            vizs = []
            for rgb, mask in zip(rgbs, masks):
                bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
                viz = imgviz.instances2rgb(
                    rgb, labels=[1], bboxes=[bbox], masks=[mask]
                )
                vizs.append(viz)
            vizs = imgviz.tile(vizs)
            imgviz.io.imsave(f'out/{index:08d}_{loss:.2f}.jpg', vizs)
            '''

    quaternion = chainer.cuda.to_cpu(quaternion.array)[0]
    translation = chainer.cuda.to_cpu(translation.array)[0]

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('quaternion, translation (upper: true, lower: pred)')
    print(examples[0]['gt_quaternion'], examples[0]['gt_translation'])
    print(quaternion, translation)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    data = examples[0]
    data['gt_quaternion'] = quaternion
    data['gt_translation'] = translation

    app = MainApp()
    app.alignment(data=data)
