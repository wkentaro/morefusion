#!/usr/bin/env python

import argparse
import json
import pathlib
import pprint

import chainer
from chainer.backends import cuda
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp


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

model = objslampp.models.MultiViewAlignmentModel(
    extractor=args_data['extractor'],
    lambda_translation=args_data['lambda_translation'],
    lambda_quaternion=args_data['lambda_quaternion'],
    loss_function=args_data.get('loss', 'l1'),
)
if args.gpu >= 0:
    model.to_gpu()

print('==> Loading trained model: {}'.format(args.model))
chainer.serializers.load_npz(args.model, model)
print('==> Done model loading')

dataset = objslampp.datasets.YCBVideoMultiViewAlignmentDataset(
    'val',
    class_ids=args_data['class_ids'],
    num_frames_scan=args_data.get('num_frames_scan'),
)

# -----------------------------------------------------------------------------

depth2rgb = imgviz.Depth2RGB()
for index in range(len(dataset)):
    examples = dataset[index:index + 1]
    inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
    with chainer.no_backprop_mode() and chainer.using_config('train', False):
        quaternion_pred, translation_pred = model.predict(
            class_id=inputs['class_id'],
            pitch=inputs['pitch'],
            cad_origin=inputs['cad_origin'],
            cad_rgbs=inputs['cad_rgbs'],
            cad_pcds=inputs['cad_pcds'],
            scan_origin=inputs['scan_origin'],
            scan_rgbs=inputs['scan_rgbs'],
            scan_pcds=inputs['scan_pcds'],
            scan_masks=inputs['scan_masks'],
        )
        translation_pred = model.xp.zeros((1, 3), dtype=np.float32)  # XXX
        translation_pred = model.translation_voxel2world(
            translation=translation_pred[0],
            scan_origin=inputs['scan_origin'][0],
            cad_origin=inputs['cad_origin'][0],
            pitch=inputs['pitch'][0],
        )
        translation_pred = cuda.to_cpu(translation_pred)
        quaternion_pred = cuda.to_cpu(quaternion_pred.array)[0]

        T_pred = tf.quaternion_matrix(quaternion_pred)
        T_pred[:3, 3] = translation_pred

        translation_true = model.translation_voxel2world(
            translation=inputs['gt_translation'][0],
            scan_origin=inputs['scan_origin'][0],
            cad_origin=inputs['cad_origin'][0],
            pitch=inputs['pitch'][0],
        )
        translation_true = cuda.to_cpu(translation_true)
        T_true = tf.quaternion_matrix(examples[0]['gt_quaternion'])
        T_true[:3, 3] = translation_true
        # T_true = examples[0]['gt_pose']

    image_id, class_id = dataset.ids[index]
    print(image_id)

    frame = dataset.get_frame(image_id)

    rgb = frame['color']
    depth = frame['depth']
    meta = frame['meta']

    # T_pred, T_true: T_cad2cam

    import trimesh

    scene = trimesh.Scene()

    cad_file = objslampp.datasets.YCBVideoModelsDataset()\
        .get_model(class_id=class_id)['textured_simple']

    import pybullet

    vizs = []
    for T in [T_true, T_pred]:
        pybullet.connect(pybullet.DIRECT)
        objslampp.extra.pybullet.add_model(
            cad_file,
            position=tf.translation_from_matrix(T),
            orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
            register=False
        )

        near = 0.01
        far = 1000.
        height = 480
        width = 640
        K = meta['intrinsic_matrix']
        fovx = np.rad2deg(2 * np.arctan(width / (2 * K[0, 0])))
        fovy = np.rad2deg(2 * np.arctan(height / (2 * K[1, 1])))
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov=fovy, aspect=1. * width / height, farVal=far, nearVal=near
        )

        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=[0, 0, 0],
            cameraTargetPosition=[0, 0, 1],
            cameraUpVector=[0, -1, 0],
        )
        _, _, onview, _, segm = pybullet.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        onview = onview[:, :, :3]
        mask = (segm == 0).astype(np.int32)

        pybullet.disconnect()

        depth_viz = depth2rgb(depth)
        import fcn
        mask_viz = fcn.utils.label2rgb(mask, img=rgb)
        viz = imgviz.tile([rgb, onview, mask_viz], (1, 3))
        vizs.append(viz)
    viz = imgviz.tile(vizs, (2, 1))

    # imgviz.io.pyglet_imshow(viz)
    # imgviz.io.pyglet_run()
    viz = imgviz.resize(viz, width=1500)
    imgviz.io.cv_imshow(viz)
    if imgviz.io.cv_waitkey() == ord('q'):
        break
