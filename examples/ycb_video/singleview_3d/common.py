import json

import chainer
from chainer.backends import cuda
import numpy as np
import path
import trimesh.transformations as tf

import morefusion
from morefusion.contrib import singleview_3d


home = path.Path('~').expanduser()
here = path.Path(__file__).abspath().parent


class Inference:

    def __init__(self, dataset='my_synthetic', gpu=0):
        # model_file = here / 'logs/20190518_022000/snapshot_model_best_auc_add.npz'  # NOQA
        model_file = here / 'logs/20190704_100641/snapshot_model_best_auc_add.npz'  # NOQA
        model_file = path.Path(model_file)
        args_file = model_file.parent / 'args'

        with open(args_file) as f:
            args_data = json.load(f)

        model = singleview_3d.models.BaselineModel(
            n_fg_class=len(args_data['class_names'][1:]),
            freeze_until=args_data['freeze_until'],
            voxelization=args_data.get('voxelization', 'average'),
        )
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            model.to_gpu()
        chainer.serializers.load_npz(model_file, model)

        if dataset == 'my_synthetic':
            root_dir = home / 'data/datasets/wkentaro/morefusion/ycb_video/synthetic_data/20190507_121544.807309'  # NOQA
            dataset = singleview_3d.datasets.MySyntheticDataset(
                root_dir=root_dir,
                class_ids=args_data['class_ids'],
            )
        elif dataset == 'my_real':
            root_dir = home / 'data/datasets/wkentaro/morefusion/ycb_video/real_data/20190614_18'  # NOQA
            dataset = singleview_3d.datasets.MyRealDataset(
                root_dir=root_dir,
                class_ids=args_data['class_ids'],
            )
        else:
            raise ValueError(f'unsupported dataset: {dataset}')

        self.gpu = gpu
        self.model = model
        self.dataset = dataset

    def __call__(self, index, bg_class=False):
        frame = self.dataset.get_frame(index, bg_class=bg_class)
        examples = self.dataset.get_examples(index)

        inputs = chainer.dataset.concat_examples(examples, device=self.gpu)

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred = self.model.predict(
                class_id=inputs['class_id'],
                pitch=inputs['pitch'],
                origin=inputs['origin'],
                rgb=inputs['rgb'],
                pcd=inputs['pcd'],
            )

        quaternion_true = cuda.to_cpu(inputs['quaternion_true'])
        translation_true = cuda.to_cpu(inputs['translation_true'])
        quaternion_pred = cuda.to_cpu(quaternion_pred.array)
        translation_pred = cuda.to_cpu(translation_pred.array)

        T_true = self.T(quaternion_true, translation_true)
        T_pred = self.T(quaternion_pred, translation_pred)

        K = frame['intrinsic_matrix']
        frame['pcd'] = morefusion.geometry.pointcloud_from_depth(
            frame['depth'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        return frame, T_true, T_pred

    @staticmethod
    def T(quaternion, translation):
        N = quaternion.shape[0]
        assert N == translation.shape[0]
        T = np.zeros((N, 4, 4), dtype=float)
        for i in range(N):
            T[i] = (
                tf.translation_matrix(translation[i]) @
                tf.quaternion_matrix(quaternion[i])
            )
        return T
