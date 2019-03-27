import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.vgg import VGG16
import numpy as np

import objslampp


class Model(chainer.Chain):

    def __init__(self, n_fg_class, freeze_until):
        super(Model, self).__init__()

        initialW = chainer.initializers.Normal(0.01)
        kwargs = {'initialW': initialW}
        with self.init_scope():
            self.extractor = VGG16(pretrained_model='imagenet')
            self.extractor.pick = ['conv4_3', 'conv3_3', 'conv2_2', 'conv1_2']
            self.extractor.remove_unused()
            self.fc5 = L.Linear(512 * 4 * 4, 4096, **kwargs)  # 512*4*4 = 8192
            self.fc6 = L.Linear(4096, 4096, **kwargs)
            self.fc_quaternion = L.Linear(4096, 4 * n_fg_class, **kwargs)

        self._n_fg_class = n_fg_class
        self._freeze_until = freeze_until

    def _get_cad_pcd(self, *, class_id):
        models = objslampp.datasets.YCBVideoModels()
        pcd_file = models.get_model(class_id=class_id)['points_xyz']
        return np.loadtxt(pcd_file)

    def evaluate(
        self,
        *,
        class_id,
        quaternion_true,
        translation_true,
        quaternion_pred,
        translation_rough,
    ):
        batch_size = class_id.shape[0]

        T_cam2cad_true = objslampp.functions.quaternion_matrix(quaternion_true)
        T_cam2cad_pred = objslampp.functions.quaternion_matrix(quaternion_pred)

        T_cam2cad_true = cuda.to_cpu(T_cam2cad_true.array)
        T_cam2cad_pred = cuda.to_cpu(T_cam2cad_pred.array)
        translation_true = cuda.to_cpu(translation_true)
        translation_rough = cuda.to_cpu(translation_rough)

        # add_rotation
        summary = chainer.DictSummary()
        for i in range(batch_size):
            class_id_i = int(class_id[i])
            cad_pcd = self._get_cad_pcd(class_id=class_id_i)
            add_rotation = objslampp.metrics.average_distance(
                [cad_pcd], [T_cam2cad_true[i]], [T_cam2cad_pred[i]]
            )[0]
            if chainer.config.train:
                summary.add({'add_rotation': add_rotation})
            else:
                summary.add({f'add_rotation/{class_id_i:04d}': add_rotation})
        chainer.report(summary.compute_mean(), self)

        T_cam2cad_true = objslampp.functions.compose_transform(
            Rs=T_cam2cad_true[:, :3, :3], ts=translation_true,
        ).array
        T_cam2cad_pred = objslampp.functions.compose_transform(
            Rs=T_cam2cad_pred[:, :3, :3], ts=translation_rough
        ).array

        # add
        summary = chainer.DictSummary()
        for i in range(batch_size):
            class_id_i = int(class_id[i])
            cad_pcd = self._get_cad_pcd(class_id=class_id_i)
            add_rotation = objslampp.metrics.average_distance(
                [cad_pcd], [T_cam2cad_true[i]], [T_cam2cad_pred[i]]
            )[0]
            if chainer.config.train:
                summary.add({'add': add_rotation})
            else:
                summary.add({f'add/{class_id_i:04d}': add_rotation})
        chainer.report(summary.compute_mean(), self)

    def predict(
        self,
        *,
        class_id,
        rgb,
    ):
        xp = self.xp

        batch_size, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3

        # prepare
        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW

        # feature extraction
        mean = xp.asarray(self.extractor.mean)
        h_conv4_3, h_conv3_3, h_conv2_2, h_conv1_2 = self.extractor(
            rgb - mean[None]
        )
        if self._freeze_until == 'conv4_3':
            h_conv4_3.unchain_backward()
        elif self._freeze_until == 'conv3_3':
            h_conv3_3.unchain_backward()
        elif self._freeze_until == 'conv2_2':
            h_conv2_2.unchain_backward()
        elif self._freeze_until == 'conv1_2':
            h_conv1_2.unchain_backward()
        else:
            self._freeze_until == 'none'
        del h_conv3_3, h_conv2_2, h_conv1_2
        h = h_conv4_3  # 1/8   # 256x256 -> 32x32
        h = F.average_pooling_2d(h, ksize=8)   # 1/64  # 256x256 -> 4x4

        # regression
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        quaternion = self.fc_quaternion(h)
        quaternion = quaternion.reshape(batch_size, self._n_fg_class, 4)
        quaternion = F.normalize(quaternion, axis=1)

        quaternion = quaternion[xp.arange(batch_size), class_id, :]

        return quaternion

    def __call__(
        self,
        *,
        class_id,
        rgb,
        quaternion_true,
        translation_true,
        translation_rough,
    ):
        quaternion_pred = self.predict(
            class_id=class_id,
            rgb=rgb,
        )

        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            quaternion_pred=quaternion_pred,
            translation_true=translation_true,
            translation_rough=translation_rough,
        )

        loss = self.loss(
            class_id=class_id,
            quaternion_true=quaternion_true,
            quaternion_pred=quaternion_pred,
        )
        return loss

    def loss(self, *, class_id, quaternion_true, quaternion_pred):
        T_cam2cad_true = objslampp.functions.quaternion_matrix(
            quaternion_true
        )
        T_cam2cad_pred = objslampp.functions.quaternion_matrix(
            quaternion_pred
        )

        batch_size = class_id.shape[0]

        loss = 0
        for i in range(batch_size):
            cad_pcd = self._get_cad_pcd(class_id=int(class_id[i]))
            cad_pcd = self.xp.asarray(cad_pcd)
            loss_i = objslampp.functions.average_distance(
                cad_pcd, T_cam2cad_true[i:i + 1], T_cam2cad_pred[i:i + 1]
            )[0]
            loss += loss_i
        loss /= batch_size

        values = {'loss': loss}
        chainer.report(values, observer=self)

        return loss
