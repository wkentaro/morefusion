#!/usr/bin/env python

import json
import logging
import pathlib
import pprint

import chainer

import objslampp
from objslampp import logger

from view_dataset import MainApp


logger.setLevel(logging.DEBUG)

# -----------------------------------------------------------------------------

gpu = 0

if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()

model = objslampp.models.SimpleMV3DCNNModel(extractor='vgg16')
if gpu >= 0:
    model.to_gpu()

# model_file = 'logs.hoop/20190312_235147.939651/snapshot_model_iter_532.npz'
model_file = 'logs.hoop/20190312_235147.939651/snapshot_model_iter_1064.npz'
# model_file = 'logs.hoop/20190312_235147.939651/snapshot_model_iter_1595.npz'
# model_file = 'logs.hoop/20190312_235147.939651/snapshot_model_iter_3190.npz'
if model_file is not None:
    logger.info('==> Loading trained model: {}'.format(model_file))
    chainer.serializers.load_npz(model_file, model)
    logger.info('==> Done model loading')

args_file = pathlib.Path(model_file).with_name('args')
with open(args_file) as f:
    data = json.load(f)
    pprint.pprint(data)

dataset = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
    # 'train',
    'val',
    class_ids=data['class_ids'],
)

# -----------------------------------------------------------------------------

examples = dataset[20:21]
inputs = chainer.dataset.concat_examples(examples, device=gpu)
inputs.pop('gt_pose')
gt_quaternion = inputs.pop('gt_quaternion')
gt_translation = inputs.pop('gt_translation')
with chainer.no_backprop_mode() and chainer.using_config('train', False):
    quaternion, translation = model.predict(**inputs)
    model.loss(
        quaternion=quaternion,
        translation=translation,
        gt_quaternion=gt_quaternion,
        gt_translation=gt_translation,
    )

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
