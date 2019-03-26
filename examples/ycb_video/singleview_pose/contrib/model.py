import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.vgg import VGG16
import numpy as np
import trimesh.transformations as tf

import objslampp


class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()

        initialW = chainer.initializers.Normal(0.01)
        with self.init_scope():
            self.extractor = VGG16(pretrained_model='imagenet')
            self.extractor.pick = ['pool4']
            self.extractor.remove_unused()
            self.fc_quaternion = L.Linear(512, 4, initialW=initialW)

    def evaluate(
        self,
        cad_pcd,
        quaternion_true,
        quaternion_pred,
        translation_true,
        translation_rough,
    ):
        assert quaternion_pred.shape[0] == 1
        cad_pcd = cuda.to_cpu(cad_pcd[0])
        quaternion_true = cuda.to_cpu(quaternion_true[0])
        quaternion_pred = cuda.to_cpu(quaternion_pred.array[0])
        translation_true = cuda.to_cpu(translation_true[0])
        translation_rough = cuda.to_cpu(translation_rough[0])

        T_cam2cad_true = tf.quaternion_matrix(quaternion_true)
        T_cam2cad_pred = tf.quaternion_matrix(quaternion_pred)
        add_rotation = objslampp.metrics.average_distance(
            [cad_pcd], [T_cam2cad_true], [T_cam2cad_pred]
        )[0]

        T_cam2cad_true = objslampp.geometry.compose_transform(
            R=T_cam2cad_true[:3, :3], t=translation_true
        )
        T_cam2cad_pred = objslampp.geometry.compose_transform(
            R=T_cam2cad_pred[:3, :3], t=translation_rough
        )
        add = objslampp.metrics.average_distance(
            [cad_pcd], [T_cam2cad_true], [T_cam2cad_pred]
        )[0]

        if chainer.config.train:
            values = {
                'add': add,
                'add_rotation': add_rotation,
            }
        else:
            values = {
                'add/0002': add,
                'add_rotation/0002': add_rotation,
            }
        chainer.report(values, observer=self)

    def predict(
        self,
        cad_pcd,
        rgb,
        quaternion_true,
        translation_true,
        translation_rough,
    ):
        xp = self.xp

        assert rgb.ndim == 4
        N, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3

        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW
        quaternion_true = quaternion_true.astype(np.float32)
        cad_pcd = cad_pcd.astype(np.float32)

        mean = xp.asarray(self.extractor.mean)
        h, = self.extractor(rgb - mean[None])  # NCHW
        h = F.average(h, axis=(2, 3))

        quaternion = F.normalize(self.fc_quaternion(h))
        return quaternion

    def __call__(
        self,
        cad_pcd,
        rgb,
        quaternion_true,
        translation_true,
        translation_rough,
    ):
        quaternion = self.predict(
            cad_pcd,
            rgb,
            quaternion_true,
            translation_true,
            translation_rough,
        )

        self.evaluate(
            cad_pcd,
            quaternion_true,
            quaternion,
            translation_true,
            translation_rough,
        )

        T_cam2cad_pred = objslampp.functions.quaternion_matrix(
            quaternion
        )
        T_cam2cad_true = objslampp.functions.quaternion_matrix(
            quaternion_true
        )

        assert cad_pcd.shape[0] == 1
        loss = objslampp.functions.average_distance(
            cad_pcd[0], T_cam2cad_true[0], T_cam2cad_pred[0]
        )

        if chainer.config.train:
            values = {'loss': loss}
        else:
            values = {'loss/0002': loss}
        chainer.report(values, observer=self)

        return loss
