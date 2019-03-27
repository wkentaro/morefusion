#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
import matplotlib.pyplot as plt
import pandas

import objslampp

from contrib import Model
from contrib import Dataset


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('model', help='model file in a log dir')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

args_file = pathlib.Path(args.model).parent / 'args'
with open(args_file) as f:
    args_data = json.load(f)
pprint.pprint(args_data)

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()

model = Model(
    n_fg_class=21,
    freeze_until='none',
)
if args.gpu >= 0:
    model.to_gpu()

print('==> Loading trained model: {}'.format(args.model))
chainer.serializers.load_npz(args.model, model)
print('==> Done model loading')

dataset = Dataset('val', class_ids=[2])

# -----------------------------------------------------------------------------

observations = []
for index in range(len(dataset)):
    image_id = dataset.ids[index][0]
    # if image_id.split('/')[0] != '0054':
    #     continue

    examples = dataset[index:index + 1]
    inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
    with chainer.no_backprop_mode() and chainer.using_config('train', False):
        quaternion_pred = model.predict(
            class_id=inputs['class_id'],
            rgb=inputs['rgb'],
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
                translation_rough=inputs['translation_rough'],
            )
        observations.append(observation)

    print(f'[{index:08d}] [{image_id}] {observation}')

df = pandas.DataFrame(observations)

errors = df['main/add_rotation/0002'].values
auc, x, y = objslampp.metrics.auc_for_errors(errors, 0.1, return_xy=True)
print('auc (add_rotation):', auc)
plt.subplot(121)
plt.title('ADD (rotation) (AUC={:.1f})'.format(auc * 100))
plt.plot(x, y)
plt.xlim(0, 0.1)
plt.ylim(0, 1)
plt.xlabel('average distance threshold [m]')
plt.ylabel('accuracy')

plt.subplot(122)
errors = df['main/add/0002'].values
auc, x, y = objslampp.metrics.auc_for_errors(errors, 0.1, return_xy=True)
print('auc (add):', auc)
plt.title('ADD (rotation + translation) (AUC={:.1f})'.format(auc * 100))
plt.plot(x, y)
plt.xlim(0, 0.1)
plt.ylim(0, 1)
plt.xlabel('average distance threshold [m]')
plt.ylabel('accuracy')

plt.tight_layout()
plt.show()
