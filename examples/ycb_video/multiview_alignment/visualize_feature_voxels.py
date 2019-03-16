#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
from chainer.backends import cuda
import imgviz
import numpy as np
import trimesh
import trimesh.viewer

import objslampp


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--model', help='model file in a log dir')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

# args.model = './logs.num_frames_scan/20190314_214815.367553/snapshot_model_iter_14885.npz'  # NOQA
args.model = './logs.num_frames_scan/20190313_204339.762084/snapshot_model_iter_12227.npz'  # NOQA

args_file = pathlib.Path(args.model).parent / 'args'
with open(args_file) as f:
    args_data = json.load(f)
pprint.pprint(args_data)

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()

model = objslampp.models.SimpleMV3DCNNModel(
    extractor=args_data['extractor'],
    lambda_translation=args_data['lambda_translation'],
    lambda_quaternion=args_data['lambda_quaternion'],
)
if args.gpu >= 0:
    model.to_gpu()

args.model = None
if args.model is not None:
    print('==> Loading trained model: {}'.format(args.model))
    chainer.serializers.load_npz(args.model, model)
    print('==> Done model loading')

dataset = objslampp.datasets.YCBVideoMultiViewAlignmentDataset(
    'val',
    class_ids=args_data['class_ids'],
    sampling=60,
    num_frames_scan=args_data.get('num_frames_scan'),
)

index = 16

examples = dataset[index:index + 1]
inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
with chainer.no_backprop_mode() and chainer.using_config('train', False):
    h_cad = model.encode(
        origin=inputs['cad_origin'][0],
        pitch=inputs['pitch'][0],
        rgbs=inputs['cad_rgbs'][0],
        pcds=inputs['cad_pcds'][0],
    )
    h_scan = model.encode(
        origin=inputs['scan_origin'][0],
        pitch=inputs['pitch'][0],
        rgbs=inputs['scan_rgbs'][0],
        pcds=inputs['scan_pcds'][0],
        masks=inputs['scan_masks'][0]
    )
    # quaternion, translation = model.predict_from_code(h_cad, h_scan)
    # loss = model.loss(
    #     quaternion=quaternion,
    #     translation=translation,
    #     gt_quaternion=inputs['gt_quaternion'],
    #     gt_translation=inputs['gt_translation'],
    # )
    #
    # quaternion = quaternion.array
    # translation = translation.array
    #
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('quaternion, translation (upper: true, lower: pred)')
    # print(inputs['gt_quaternion'], inputs['gt_translation'])
    # print(quaternion, translation)
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #
    # loss = float(loss.array)
    # video_id = int(inputs['video_id'][0])
    # print(f'[{video_id:04d}] [{index:08d}] {loss}')

nchannel2rgb = imgviz.Nchannel2RGB()
for h_cad in [h_cad, h_scan]:
    h_cad = cuda.to_cpu(h_cad.array)
    voxel = h_cad[0].transpose(1, 2, 3, 0)
    np.set_printoptions(formatter={'float': '{:.3f}'.format}, linewidth=100)
    print(voxel.shape)
    mask = ~(voxel == 0).all(axis=3)
    print(voxel[mask].shape)
    print(voxel[mask].min(axis=0))
    print(voxel[mask].mean(axis=0))
    print(voxel[mask].max(axis=0))
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

    def show(scene, **kwargs):

        resolution = kwargs.pop('resolution', (500, 500))

        def callback(scene):
            if hasattr(scene, 'angles'):
                scene.angles += [0, np.deg2rad(1), 0]
            else:
                scene.angles = np.zeros(3)
            scene.set_camera(angles=scene.angles)

        return trimesh.viewer.SceneViewer(
            scene=scene, callback=callback, resolution=resolution, **kwargs
        )

    show(scene=scene)
