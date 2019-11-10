#!/usr/bin/env python

import argparse
import json
import re

import chainer
import pandas
import path

import objslampp
from objslampp.contrib import singleview_3d as contrib

from train import Transform


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('log_dir', help='log dir')
args = parser.parse_args()

args.log_dir = path.Path(args.log_dir)

with open(args.log_dir / 'args') as f:
    args_dict = json.load(f)

model = contrib.models.Model(
    n_fg_class=len(args_dict['class_names'][1:]),
    pretrained_resnet18=args_dict['pretrained_resnet18'],
    with_occupancy=args_dict['with_occupancy'],
)
chainer.serializers.load_npz(
    args.log_dir / 'snapshot_model_best_add.npz', model
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

class_names = objslampp.datasets.ycb_video.class_names
data = {'class_id': [], 'add_or_add_s_2cm': [], 'add_or_add_s': [], 'add_s': []}
for cls_id, _ in enumerate(class_names):
    if cls_id == 0:
        continue
    data['class_id'].append(cls_id)
    data['add_or_add_s_2cm'].append(result[f'validation/main/<2cm/add_or_add_s/{cls_id:04d}'])
    data['add_or_add_s'].append(result[f'validation/main/auc/add_or_add_s/{cls_id:04d}'])
    data['add_s'].append(result[f'validation/main/auc/add_s/{cls_id:04d}'])
df = pandas.DataFrame(data)

print(df)
df.to_csv(args.log_dir / 'eval_result.csv')

import IPython; IPython.embed()  # NOQA
