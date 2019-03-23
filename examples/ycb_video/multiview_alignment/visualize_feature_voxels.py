#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
from chainer.backends import cuda
import imgviz
import numpy as np
import pyglet
import trimesh

import objslampp


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('model', help='model file in a log dir')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--show', action='store_true', help='show visualization')
parser.add_argument('--index', nargs='+', type=int, help='dataset indices')
args = parser.parse_args()

args_file = pathlib.Path(args.model).parent / 'args'
with open(args_file) as f:
    args_data = json.load(f)
pprint.pprint(args_data)

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()

model = objslampp.models.MultiViewAlignmentModel(
    extractor=args_data['extractor'],
    lambda_translation=args_data['lambda_translation'],
    lambda_quaternion=args_data['lambda_quaternion'],
)
if args.gpu >= 0:
    model.to_gpu()

print('==> Loading trained model: {}'.format(args.model))
chainer.serializers.load_npz(args.model, model)
print('==> Done model loading')

dataset = objslampp.datasets.YCBVideoMultiViewAlignmentDataset(
    'val',
    class_ids=args_data['class_ids'],
    sampling=60,
    num_frames_scan=args_data.get('num_frames_scan'),
)

# -----------------------------------------------------------------------------

if not args.index:
    args.index = range(len(dataset))

for index in args.index:
    examples = dataset[index:index + 1]
    inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
    with chainer.no_backprop_mode() and chainer.using_config('train', False):
        h_cad = model.encode(
            origin=inputs['cad_origin'][0],
            pitch=inputs['pitch'][0],
            rgbs=inputs['cad_rgbs'][0],
            pcds=inputs['cad_pcds'][0],
            return_fused=args.show,
        )
        h_scan = model.encode(
            origin=inputs['scan_origin'][0],
            pitch=inputs['pitch'][0],
            rgbs=inputs['scan_rgbs'][0],
            pcds=inputs['scan_pcds'][0],
            masks=inputs['scan_masks'][0],
            return_fused=args.show,
        )

        try:
            quaternion, translation = model.predict_from_code(h_cad, h_scan)
            loss = model.loss(
                quaternion=quaternion,
                translation=translation,
                gt_quaternion=inputs['gt_quaternion'],
                gt_translation=inputs['gt_translation'],
            )

            quaternion = quaternion.array
            translation = translation.array

            loss = float(loss.array)
            video_id = int(inputs['video_id'][0])
            print(f'[{video_id:04d}] [{index:08d}] {loss}')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('quaternion, translation (upper: true, lower: pred)')
            print(inputs['gt_quaternion'], inputs['gt_translation'])
            print(quaternion, translation)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        except chainer.utils.type_check.InvalidType:
            pass

    if not args.show:
        del h_cad, h_scan, quaternion, translation, loss, examples, inputs
        continue

    nchannel2rgb = imgviz.Nchannel2RGB()
    for name, h_cad in zip(['cad', 'scan'], [h_cad, h_scan]):
        h_cad = cuda.to_cpu(h_cad.array)
        voxel = h_cad[0].transpose(1, 2, 3, 0)
        np.set_printoptions(
            formatter={'float': '{:.3f}'.format}, linewidth=200
        )
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('voxel    :', voxel.shape)
        mask = ~(voxel == 0).all(axis=3)
        print('voxel > 0:', voxel[mask].shape)
        print('min      :', voxel[mask].min(axis=0))
        print('mean     :', voxel[mask].mean(axis=0))
        print('max      :', voxel[mask].max(axis=0))
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        X, Y, Z, C = voxel.shape

        if C == 3:
            colors = voxel.reshape(X, Y, Z, 3).astype(np.uint8)
        else:
            colors = nchannel2rgb(voxel.reshape(1, -1, C))
            colors = colors.reshape(X, Y, Z, 3)

        matrix = (voxel != 0).any(axis=3)
        geom = trimesh.voxel.Voxel(
            matrix=matrix,
            pitch=float(inputs['pitch'][0]),
            origin=cuda.to_cpu(inputs['cad_origin'][0]),
        )
        geom = geom.as_boxes()
        I, J, K = zip(*np.argwhere(matrix))
        geom.visual.face_colors = \
            colors[I, J, K].repeat(12, axis=0)

        scene = trimesh.Scene(geom)
        objslampp.vis.trimesh.show_with_rotation(
            scene=scene,
            resolution=(500, 500),
            caption=name,
            start_loop=False,
        )
    pyglet.app.run()
