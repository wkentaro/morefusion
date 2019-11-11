#!/usr/bin/env python

import argparse
import json

import chainer
import numpy as np
import path
import trimesh

import objslampp

import contrib


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
    centerize_pcd=args_dict['centerize_pcd'],
    pretrained_resnet18=args_dict['pretrained_resnet18'],
)
chainer.serializers.load_npz(
    args.log_dir / 'snapshot_model_best_add.npz', model
)
model.to_gpu(0)

dataset = objslampp.datasets.MySyntheticYCB20190916RGBDPoseEstimationDataset(
    split='val',
    class_ids=args_dict['class_ids'],
)

models = objslampp.datasets.YCBVideoModels()
for index in range(len(dataset)):
    frame = dataset.get_frame(index)
    examples = dataset.get_example(index)

    batch = chainer.dataset.concat_examples(examples, device=0)
    quaternion, translation, confidence = model.predict(
        class_id=batch['class_id'],
        rgb=batch['rgb'],
        pcd=batch['pcd'],
    )
    quaternion = chainer.cuda.to_cpu(quaternion.array)
    translation = chainer.cuda.to_cpu(translation.array)
    confidence = chainer.cuda.to_cpu(confidence.array)
    indices = np.argmax(confidence, axis=1)
    quaternion = quaternion[np.arange(len(examples)), indices]
    translation = translation[np.arange(len(examples)), indices]

    transform = objslampp.functions.transformation_matrix(
        quaternion, translation,
    ).array
    transform_true = objslampp.functions.transformation_matrix(
        batch['quaternion_true'], batch['translation_true']
    ).array
    transform_true = chainer.cuda.to_cpu(transform_true)

    scene = trimesh.Scene()
    scene_true = trimesh.Scene(camera=scene.camera)
    for i in range(len(examples)):
        class_id = examples[i]['class_id']
        cad = models.get_cad(class_id)
        if hasattr(cad.visual, 'to_color'):
            cad.visual = cad.visual.to_color()
        scene.add_geometry(cad, transform=transform[i])
        scene_true.add_geometry(cad, transform=transform_true[i])
    scene.camera.transform = objslampp.extra.trimesh.to_opengl_transform()
    scenes = {'pose': scene, 'pose_true': scene_true, 'rgb': frame['rgb']}
    objslampp.extra.trimesh.display_scenes(scenes, tile=(1, 3))

    break
