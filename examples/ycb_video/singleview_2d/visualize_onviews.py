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

from lib import Model
from lib import Dataset


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

model = Model()
if args.gpu >= 0:
    model.to_gpu()

print('==> Loading trained model: {}'.format(args.model))
chainer.serializers.load_npz(args.model, model)
print('==> Done model loading')

dataset = Dataset('val', class_ids=[2])

# -----------------------------------------------------------------------------

depth2rgb = imgviz.Depth2RGB()
for index in range(len(dataset)):
    examples = dataset[index:index + 1]
    inputs = chainer.dataset.concat_examples(examples, device=args.gpu)
    cad_pcd, rgb, quaternion_true, translation_true, translation_rough = inputs
    with chainer.no_backprop_mode() and chainer.using_config('train', False):
        quaternion_pred = model.predict(*inputs)
        quaternion_pred = cuda.to_cpu(quaternion_pred.array)[0]
        translation_pred = cuda.to_cpu(translation_rough)[0]
        T_pred = tf.quaternion_matrix(quaternion_pred)
        T_pred[:3, 3] = translation_pred

        translation_true = cuda.to_cpu(translation_true)[0]
        quaternion_true = cuda.to_cpu(quaternion_true)[0]
        T_true = tf.quaternion_matrix(quaternion_true)
        T_true[:3, 3] = translation_true

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
        mask_viz = imgviz.label2rgb(mask, img=rgb, alpha=0.7)
        viz = imgviz.tile([rgb, onview, mask_viz], (1, 3))
        vizs.append(viz)
    viz = imgviz.tile(vizs, (2, 1))

    # imgviz.io.pyglet_imshow(viz)
    # imgviz.io.pyglet_run()
    viz = imgviz.resize(viz, width=1500)
    # imgviz.io.cv_imshow(viz)
    # if imgviz.io.cv_waitkey() == ord('q'):
    #     break
    image_id = dataset.ids[index][0]
    out_file = pathlib.Path(f'out/{image_id}.jpg')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    imgviz.io.imsave(out_file, viz)
