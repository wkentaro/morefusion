#!/usr/bin/env python

import argparse
import collections
import concurrent.futures

import imgviz
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import trimesh.transformations as tf

import objslampp

import contrib


def get_adds(result_file):
    dataset = objslampp.datasets.YCBVideoDataset
    models = objslampp.datasets.YCBVideoModels()

    result = scipy.io.loadmat(
        result_file, chars_as_strings=True, squeeze_me=True
    )
    frame_id = '/'.join(result['frame_id'].split('/')[1:])

    frame = dataset.get_frame(frame_id)

    adds = collections.defaultdict(list)
    for gt_index, cls_id in enumerate(frame['meta']['cls_indexes']):
        try:
            pred_index = np.where(result['labels'] == cls_id)[0][0]
            pose = result['poses'][pred_index]
            T_pred = objslampp.geometry.compose_transform(
                R=tf.quaternion_matrix(pose[:4])[:3, :3],
                t=pose[4:],
            )
            T_true = np.r_[
                frame['meta']['poses'][:, :, gt_index],
                [[0, 0, 0, 1]],
            ]
            pcd_file = models.get_pcd_model(class_id=cls_id)
            pcd = np.loadtxt(pcd_file)
            add = objslampp.metrics.average_distance(
                [pcd],
                transform1=[T_true],
                transform2=[T_pred],
            )[0]
        except IndexError:
            add = np.inf
        print(f'{result_file}, {cls_id:04d}, {add:.3f}')
        adds[cls_id].append(add)
    return adds


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--name',
        choices=['Densefusion_iterative_result', 'Densefusion_icp_result'],
        default='Densefusion_iterative_result',
    )
    args = parser.parse_args()

    result_dir = contrib.get_eval_result(name=args.name)

    adds_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result_file in result_dir.glob('*.mat'):
            adds = executor.submit(get_adds, result_file)
            adds_list.append(adds)

    adds_all = collections.defaultdict(list)
    for future in adds_list:
        adds = future.result()
        for cls_id in adds:
            adds_all[cls_id] += adds[cls_id]

    for cls_id, adds in sorted(adds_all.items()):
        class_name = objslampp.datasets.ycb_video.class_names[cls_id]

        auc, x, y = objslampp.metrics.ycb_video_add_auc(
            adds, max_value=0.1, return_xy=True)

        fig = plt.figure(figsize=(10, 5))

        print('auc (add):', auc)
        plt.title(f'{class_name}: ADD (AUC={auc * 100:.2f})')
        plt.plot(x, y, color='b')
        plt.xlim(0, 0.1)
        plt.ylim(0, 1)
        plt.xlabel('average distance threshold [m]')
        plt.ylabel('accuracy')

        plt.tight_layout()
        img = imgviz.io.pyplot_fig2arr(fig)
        plt.close()
        out_file = f'plots/{class_name}.png'
        print('==> Saved ADD curve plot:', out_file)
        imgviz.io.imsave(out_file, img)


if __name__ == '__main__':
    main()
