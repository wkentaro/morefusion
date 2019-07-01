import collections

import numpy as np
import scipy
import trimesh.transformations as tf

import objslampp


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
            pcd_file = models.get_pcd_file(class_id=cls_id)
            pcd = np.loadtxt(pcd_file)
            add, add_s = objslampp.metrics.average_distance(
                [pcd],
                transform1=[T_true],
                transform2=[T_pred],
            )
            add = add[0]
            add_s = add_s[0]
        except IndexError:
            add = np.inf
            add_s = np.inf
        print(f'{result_file}, {cls_id:04d}, {add:.3f}, {add_s:.3f}')
        adds[cls_id].append((add, add_s))
    return adds
