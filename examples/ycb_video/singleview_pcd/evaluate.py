#!/usr/bin/env python

import argparse
import json

import chainer
import cupy
import numpy as np
import pandas
import path
import pybullet  # NOQA
import tqdm

import morefusion

import contrib


models = morefusion.datasets.YCBVideoModels()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args = parser.parse_args()

    args.log_dir = path.Path(
        './data.gdrive/logs.20191008.all_data/20191014_092228.858011713'
    )

    with open(args.log_dir / 'args') as f:
        args_dict = json.load(f)

    model = contrib.models.Model(
        n_fg_class=len(args_dict['class_names'][1:]),
        centerize_pcd=args_dict['centerize_pcd'],
        pretrained_resnet18=args_dict['pretrained_resnet18'],
    )
    assert args_dict['pretrained_resnet18'] is True
    assert args_dict['centerize_pcd'] is True
    chainer.serializers.load_npz(
        args.log_dir / 'snapshot_model_best_add.npz', model
    )
    model.to_gpu(0)

    dataset = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDataset(  # NOQA
        split='val',
        class_ids=args_dict['class_ids'],
    )

    def transform(examples):
        for example in examples:
            grid_target = example.pop('grid_target') > 0.5
            grid_nontarget = example.pop('grid_nontarget') > 0.5
            grid_empty = example.pop('grid_empty') > 0.5
            grid_nontarget_empty = grid_nontarget | grid_empty
            example['grid_target'] = grid_target
            example['grid_nontarget_empty'] = grid_nontarget_empty
        return examples

    dataset = chainer.datasets.TransformDataset(dataset, transform)

    data = []
    for index in tqdm.trange(len(dataset)):
        if (index + 1) % 5 != 0:
            continue

        examples = dataset.get_example(index)

        batch = chainer.dataset.concat_examples(examples, device=0)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            quaternion, translation, confidence = model.predict(
                class_id=batch['class_id'],
                rgb=batch['rgb'],
                pcd=batch['pcd'],
            )
        indices = model.xp.argmax(confidence.array, axis=1)
        quaternion = quaternion[model.xp.arange(len(examples)), indices]
        translation = translation[model.xp.arange(len(examples)), indices]

        transform = morefusion.functions.transformation_matrix(
            chainer.cuda.to_cpu(quaternion.array),
            chainer.cuda.to_cpu(translation.array),
        ).array

        transform_true = morefusion.functions.transformation_matrix(
            batch['quaternion_true'], batch['translation_true']
        ).array
        transform_true = chainer.cuda.to_cpu(transform_true)

        # visualization
        '''
        import trimesh
        frame = dataset._dataset.get_frame(index)
        scene = trimesh.Scene()
        scene_true = trimesh.Scene(camera=scene.camera)
        for i in range(len(examples)):
            class_id = examples[i]['class_id']
            cad = models.get_cad(class_id)
            if hasattr(cad.visual, 'to_color'):
                cad.visual = cad.visual.to_color()
            scene.add_geometry(cad, transform=transform[i])
            scene_true.add_geometry(cad, transform=transform_true[i])
        scene.camera.transform = morefusion.extra.trimesh.to_opengl_transform()
        scenes = {'pose': scene, 'pose_true': scene_true, 'rgb': frame['rgb']}
        morefusion.extra.trimesh.display_scenes(scenes, tile=(1, 3))
        '''

        # add result w/ occupancy
        for i in range(len(examples)):
            points = models.get_pcd(class_id=examples[i]['class_id'])
            add, add_s = morefusion.metrics.average_distance(
                [points], transform_true[i:i + 1], transform[i:i + 1]
            )
            add, add_s = add[0], add_s[0]
            if examples[i]['class_id'] in morefusion.datasets.ycb_video.class_ids_symmetric:  # NOQA
                add_or_add_s = add_s
            else:
                add_or_add_s = add
            data.append({
                'frame_index': index,
                'batch_index': i,
                'class_id': examples[i]['class_id'],
                'add_or_add_s': add_or_add_s,
                'add_s': add_s,
                'visibility': examples[i]['visibility'],
                'method': 'densefusion',
            })

        # transform_icp = iterative_closest_point(examples, batch, transform)
        #
        # for i in range(len(examples)):
        #     points = models.get_pcd(class_id=examples[i]['class_id'])
        #     add, add_s = morefusion.metrics.average_distance(
        #         [points], transform_true[i:i + 1], transform_icp[i:i + 1]
        #     )
        #     add, add_s = add[0], add_s[0]
        #     if examples[i]['class_id'] in morefusion.datasets.ycb_video.class_ids_symmetric:  # NOQA
        #         add_or_add_s = add_s
        #     else:
        #         add_or_add_s = add
        #     data.append({
        #         'frame_index': index,
        #         'batch_index': i,
        #         'class_id': examples[i]['class_id'],
        #         'add_or_add_s': add_or_add_s,
        #         'add_s': add_s,
        #         'visibility': examples[i]['visibility'],
        #         'method': 'densefusion+icp',
        #     })
        #
        # transform_icc = iterative_collision_check(examples, batch, transform)
        #
        # for i in range(len(examples)):
        #     points = models.get_pcd(class_id=examples[i]['class_id'])
        #     add, add_s = morefusion.metrics.average_distance(
        #         [points], transform_true[i:i + 1], transform_icc[i:i + 1]
        #     )
        #     add, add_s = add[0], add_s[0]
        #     if examples[i]['class_id'] in morefusion.datasets.ycb_video.class_ids_symmetric:  # NOQA
        #         add_or_add_s = add_s
        #     else:
        #         add_or_add_s = add
        #     data.append({
        #         'frame_index': index,
        #         'batch_index': i,
        #         'class_id': examples[i]['class_id'],
        #         'add_or_add_s': add_or_add_s,
        #         'add_s': add_s,
        #         'visibility': examples[i]['visibility'],
        #         'method': 'densefusion+icc',
        #     })
        #
        # transform_icc_icp = iterative_closest_point(
        #     examples, batch, transform_icc, n_iteration=30
        # )
        #
        # for i in range(len(examples)):
        #     points = models.get_pcd(class_id=examples[i]['class_id'])
        #     add, add_s = morefusion.metrics.average_distance(
        #         [points], transform_true[i:i + 1], transform_icc_icp[i:i + 1]
        #     )
        #     add, add_s = add[0], add_s[0]
        #     if examples[i]['class_id'] in morefusion.datasets.ycb_video.class_ids_symmetric:  # NOQA
        #         add_or_add_s = add_s
        #     else:
        #         add_or_add_s = add
        #     data.append({
        #         'frame_index': index,
        #         'batch_index': i,
        #         'class_id': examples[i]['class_id'],
        #         'add_or_add_s': add_or_add_s,
        #         'add_s': add_s,
        #         'visibility': examples[i]['visibility'],
        #         'method': 'densefusion+icc+icp',
        #     })

        if (index + 1) % 15 == 0:
            df = pandas.DataFrame(data)
            df.to_csv(f'data.{index:08d}.csv')


def iterative_closest_point(examples, batch, transform, n_iteration=100):
    transform_icp = []
    for i in range(len(examples)):
        nonnan = ~np.isnan(examples[i]['pcd']).any(axis=2)
        icp = morefusion.contrib.ICPRegistration(
            examples[i]['pcd'][nonnan],
            models.get_pcd(class_id=examples[i]['class_id']),
            transform[i],
        )
        transform_i = icp.register(iteration=n_iteration)
        transform_icp.append(transform_i)
    return np.array(transform_icp, dtype=np.float32)


def iterative_collision_check(examples, batch, transform):
    # refine with occupancy
    link = morefusion.contrib.CollisionBasedPoseRefinementLink(
        transform,
    )
    link.to_gpu()
    optimizer = chainer.optimizers.Adam(alpha=0.01)
    optimizer.setup(link)
    link.translation.update_rule.hyperparam.alpha *= 0.1
    #
    points = []
    sdfs = []
    for i in range(len(examples)):
        pcd, sdf = models.get_sdf(examples[i]['class_id'])
        keep = ~np.isnan(sdf)
        pcd, sdf = pcd[keep], sdf[keep]
        points.append(cupy.asarray(pcd, dtype=np.float32))
        sdfs.append(cupy.asarray(sdf, dtype=np.float32))
    #
    for i in range(30):
        loss = link(
            points,
            sdfs,
            batch['pitch'].astype(np.float32),
            batch['origin'].astype(np.float32),
            batch['grid_target'].astype(np.float32),
            batch['grid_nontarget_empty'].astype(np.float32),
        )
        loss.backward()
        optimizer.update()
        link.zerograds()
    #
    transform = morefusion.functions.transformation_matrix(
        link.quaternion, link.translation
    ).array
    transform = chainer.cuda.to_cpu(transform)
    return transform


if __name__ == '__main__':
    main()
