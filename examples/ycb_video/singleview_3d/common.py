import json
import pathlib

import chainer
from chainer.backends import cuda
import numpy as np
import trimesh.transformations as tf

import contrib


class Inference:

    def __init__(self, gpu=0):
        model_file = './logs.20190417.cad_only/20190412_142459.904281/snapshot_model_best_auc_add.npz'  # NOQA
        model_file = pathlib.Path(model_file)
        args_file = model_file.parent / 'args'

        class_ids = [2]
        root_dir = '~/data/datasets/wkentaro/objslampp/ycb_video/synthetic_data/20190428_165745.028250'  # NOQA
        root_dir = pathlib.Path(root_dir).expanduser()

        with open(args_file) as f:
            args_data = json.load(f)

        model = contrib.models.BaselineModel(
            freeze_until=args_data['freeze_until'],
            voxelization=args_data.get('voxelization', 'average'),
        )
        if gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            model.to_gpu()
        chainer.serializers.load_npz(model_file, model)

        dataset = contrib.datasets.MySyntheticDataset(
            root_dir=root_dir,
            class_ids=class_ids,
        )

        self.gpu = gpu
        self.model = model
        self.dataset = dataset

    def __call__(self, index):
        frame = self.dataset.get_frame(index)
        examples = self.dataset.get_examples(index)

        inputs = chainer.dataset.concat_examples(examples, device=self.gpu)

        with chainer.no_backprop_mode() and \
                chainer.using_config('train', False):
            quaternion_pred, translation_pred = self.model.predict(
                class_id=inputs['class_id'],
                pitch=inputs['pitch'],
                rgb=inputs['rgb'],
                pcd=inputs['pcd'],
            )

        quaternion_true = cuda.to_cpu(inputs['quaternion_true'])
        translation_true = cuda.to_cpu(inputs['translation_true'])
        quaternion_pred = cuda.to_cpu(quaternion_pred.array)
        translation_pred = cuda.to_cpu(translation_pred.array)

        T_true = self.T(quaternion_true, translation_true)
        T_pred = self.T(quaternion_pred, translation_pred)

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
