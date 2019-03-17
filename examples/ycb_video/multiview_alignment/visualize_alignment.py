#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
import numpy as np
import trimesh

import objslampp

from view_dataset import MainApp


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--model', required=True, help='model file in a log dir')
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

    if not args.show:
        continue

    data = examples[0]
    translation = data['gt_translation']  # use gt

    def show(scene, **kwargs):

        resolution = kwargs.pop('resolution', (400, 400))

        def callback(scene):
            if hasattr(scene, 'angles'):
                scene.angles += [0, np.deg2rad(1), 0]
            else:
                scene.angles = np.zeros(3)
            scene.set_camera(angles=scene.angles)

        return trimesh.viewer.SceneViewer(
            scene=scene, callback=callback, resolution=resolution, **kwargs
        )

    pred_pose = trimesh.transformations.quaternion_matrix(quaternion)
    pred_pose[:3, 3] = (
        (data['scan_origin'] - data['cad_origin']) +
        (translation * 32 * data['pitch'])
    )

    models = objslampp.datasets.YCBVideoModelsDataset()
    model_file = models.get_model(class_id=data['class_id'])

    app = MainApp()
    map_scan = app.scan_voxel_mapping(data=data, show=False)
    geom = map_scan.as_boxes()
    bbox = map_scan.as_bbox(edge=True)

    scene = trimesh.Scene()
    scene.add_geometry(geom)
    scene.add_geometry(bbox)
    show(scene, caption='scan', start_loop=False)

    cad_true = trimesh.load(str(model_file['textured_simple']))
    cad_true.visual = cad_true.visual.to_color()
    origin = trimesh.creation.icosphere(radius=0.01)
    origin.visual.face_colors = (1.0, 0., 0.)
    cad_true.apply_transform(data['gt_pose'])
    scene_true = scene.copy()
    scene_true.add_geometry(cad_true)
    show(scene_true, caption='true', start_loop=False)

    scene_pred = scene.copy()
    cad_pred = trimesh.load(str(model_file['textured_simple']))
    cad_pred.visual = cad_pred.visual.to_color()
    cad_pred.apply_transform(pred_pose)
    scene_pred.add_geometry(cad_pred)
    show(scene_pred, caption='pred', start_loop=False)

    scene = trimesh.Scene()
    cad_true.visual.face_colors = (0, 1.0, 0, 0.5)
    cad_pred.visual.face_colors = (0, 0, 1.0, 0.5)
    scene.add_geometry(cad_true)
    scene.add_geometry(cad_pred)
    axis = trimesh.creation.axis(origin_size=0.02)
    axis.apply_transform(data['gt_pose'])
    scene.add_geometry(axis)
    show(scene, caption=f'true & pred {loss:.3g}', start_loop=False)

    import pyglet
    pyglet.app.run()

    if 0:
        data = examples[0]
        data['gt_quaternion'] = quaternion
        data['gt_translation'] = translation
        app = MainApp()
        app.alignment(data=data)
