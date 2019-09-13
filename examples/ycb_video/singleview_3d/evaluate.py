#!/usr/bin/env python

import json

import chainer
import path

import objslampp

import contrib

from train import Transform


log_dir = 'logs.20190901/20190912_103821.380887599'
log_dir = path.Path(log_dir)

with open(log_dir / 'args') as f:
    args_dict = json.load(f)

model = contrib.models.Model(
    n_fg_class=len(args_dict['class_names'][1:]),
    pretrained_resnet18=args_dict['pretrained_resnet18'],
)
chainer.serializers.load_npz(
    log_dir / 'snapshot_model_best_auc_add.npz', model
)
model.to_gpu(0)

dataset = objslampp.datasets.YCBVideoPoseCNNResultsRGBDPoseEstimationDatasetReIndexed(  # NOQA
    class_ids=args_dict['class_ids'],
)
dataset = chainer.datasets.TransformDataset(
    dataset,
    Transform(train=False, with_occupancy=args_dict['with_occupancy']),
)
iterator = chainer.iterators.MultiprocessIterator(
    dataset, batch_size=16, repeat=False, shuffle=False
)

evaluator = objslampp.training.extensions.PoseEstimationEvaluator(
    iterator=iterator,
    target=model,
    device=0,
    progress_bar=True,
)
evaluator.name = evaluator.default_name
result = evaluator()

import IPython; IPython.embed()  # NOQA
