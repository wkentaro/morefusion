import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

import objslampp

from .pspnet import PSPNetExtractor


class BaselineModel(chainer.Chain):

    _models = objslampp.datasets.YCBVideoModels()
    _voxel_dim = 32
    _n_point = 1000
    _lambda_confidence = 0.015

    def __init__(
        self,
        *,
        n_fg_class,
        loss=None,
        loss_scale=None,
    ):
        super().__init__()

        self._n_fg_class = n_fg_class

        self._loss = 'add/add_s' if loss is None else loss
        assert self._loss in [
            'add',
            'add_s',
            'add/add_s',
            'add+add_s',
        ]

        if loss_scale is None:
            loss_scale = {
                'add+add_s': 0.5,
            }
        self._loss_scale = loss_scale

        with self.init_scope():
            # extractor
            self.resnet_extractor = objslampp.models.ResNet18()
            self.pspnet_extractor = PSPNetExtractor()
            self.voxel_extractor = VoxelFeatureExtractor()

            self.conv1_rot = L.Convolution1D(1024, 640, 1)
            self.conv1_trans = L.Convolution1D(1024, 640, 1)
            self.conv1_conf = L.Convolution1D(1024, 640, 1)
            self.conv2_rot = L.Convolution1D(640, 256, 1)
            self.conv2_trans = L.Convolution1D(640, 256, 1)
            self.conv2_conf = L.Convolution1D(640, 256, 1)
            self.conv3_rot = L.Convolution1D(256, 128, 1)
            self.conv3_trans = L.Convolution1D(256, 128, 1)
            self.conv3_conf = L.Convolution1D(256, 128, 1)
            self.conv4_rot = L.Convolution1D(128, n_fg_class * 4, 1)
            self.conv4_trans = L.Convolution1D(128, n_fg_class * 3, 1)
            self.conv4_conf = L.Convolution1D(128, n_fg_class, 1)

    def predict(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true=None,
        translation_true=None,
    ):
        values, points = self._extract(
            rgb=rgb,
            pcd=pcd,
        )

        pitch, origin, voxelized, count = self._voxelize(
            class_id=class_id,
            values=values,
            points=points,
        )

        quaternion_pred, translation_pred, confidence_pred = \
            self._predict_from_voxelized(
                class_id=class_id,
                pitch=pitch,
                origin=origin,
                voxelized=voxelized,
                count=count,
            )

        return quaternion_pred, translation_pred, confidence_pred

    def _extract(self, rgb, pcd):
        xp = self.xp

        B, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3

        # prepare
        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # BHWC -> BCHW
        pcd = pcd.transpose(0, 3, 1, 2).astype(np.float32)  # BHW3 -> B3HW

        # feature extraction
        mean = xp.asarray(self.resnet_extractor.mean)
        h_rgb = self.resnet_extractor(rgb - mean[None])  # 1/8
        h_rgb = self.pspnet_extractor(h_rgb)  # 1/1

        mask = ~xp.isnan(pcd).any(axis=1)  # BHW
        values = []  # BMC
        points = []  # BM3
        for i in range(B):
            # mask[i]:  HW
            # h_rgb[i]: CHW
            # pcd[i]:   3HW
            values.append(h_rgb[i][:, mask[i]].transpose(1, 0))  # MC
            points.append(pcd[i][:, mask[i]].transpose(1, 0))    # M3

        return values, points

    def _voxelize(
        self,
        class_id,
        values,
        points,
    ):
        xp = self.xp

        B = class_id.shape[0]
        dimensions = (self._voxel_dim,) * 3

        pitch = xp.empty((B,), dtype=np.float32)
        origin = xp.empty((B, 3), dtype=np.float32)

        voxelized = []
        count = []
        for i in range(B):
            pitch[i] = self._models.get_voxel_pitch(
                dimension=self._voxel_dim, class_id=int(class_id[i]),
            )
            if xp == np:
                center_i = np.median(points[i], axis=0)
            else:
                center_i = objslampp.extra.cupy.median(points[i], axis=0)
            origin[i] = center_i - pitch[i] * (self._voxel_dim / 2. - 0.5)
            voxelized_i, count_i = objslampp.functions.average_voxelization_3d(
                values=values[i],
                points=points[i],
                origin=origin[i],
                pitch=pitch[i],
                dimensions=dimensions,
                return_counts=True,
            )
            voxelized.append(voxelized_i)
            count.append(count_i)
        voxelized = F.stack(voxelized)
        count = xp.stack(count)

        return pitch, origin, voxelized, count

    def _predict_from_voxelized(
        self, class_id, pitch, origin, voxelized, count
    ):
        xp = self.xp
        B = class_id.shape[0]

        h = self.voxel_extractor(voxelized)

        values = []
        points = []
        for i in range(B):
            I, J, K = xp.nonzero(count[i])
            n_point = len(I)
            if n_point >= self._n_point:
                keep = xp.random.permutation(n_point)[:self._n_point]
            else:
                keep = xp.r_[
                    xp.arange(n_point),
                    xp.random.randint(0, n_point, self._n_point - n_point)
                ]
            assert keep.shape == (self._n_point,)
            I, J, K = I[keep], J[keep], K[keep]
            values.append(h[i, I, J, K])
            points.append(xp.column_stack((I, J, K)))
        values = F.stack(values)
        points = xp.stack(points)
        points = origin[:, None, :] + points * pitch[:, None, None]

        h = values

        h_rot = F.relu(self.conv1_rot(h))
        h_trans = F.relu(self.conv1_trans(h))
        h_conf = F.relu(self.conv1_conf(h))
        h_rot = F.relu(self.conv2_rot(h_rot))
        h_trans = F.relu(self.conv2_trans(h_trans))
        h_conf = F.relu(self.conv2_conf(h_conf))
        h_rot = F.relu(self.conv3_rot(h_rot))
        h_trans = F.relu(self.conv3_trans(h_trans))
        h_conf = F.relu(self.conv3_conf(h_conf))
        cls_rot = self.conv4_rot(h_rot)
        cls_trans = self.conv4_trans(h_trans)
        cls_conf = self.conv4_conf(h_conf)

        quaternion = cls_rot.reshape(B, self._n_fg_class, 4)
        translation = cls_trans.reshape(B, self._n_fg_class, 3)
        confidence = cls_conf.reshape(B, self._n_fg_class)

        fg_class_id = class_id - 1
        quaternion = quaternion[xp.arange(B), fg_class_id, :]
        translation = translation[xp.arange(B), fg_class_id, :]
        confidence = confidence[xp.arange(B), fg_class_id]

        quaternion = F.normalize(quaternion, axis=1)
        translation = points + translation * pitch[:, None]

        return quaternion, translation, confidence

    def __call__(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        B = class_id.shape[0]
        xp = self.xp

        quaternion_pred, translation_pred, confidence_pred = self.predict(
            class_id=class_id,
            rgb=rgb,
            pcd=pcd,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )

        indices = F.argmax(confidence_pred, axis=1).array
        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred[xp.arange(B), indices],
            translation_pred=translation_pred[xp.arange(B), indices],
        )

        loss = self.loss(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
            confidence_pred=confidence_pred,
        )
        return loss

    def evaluate(
        self,
        *,
        class_id,
        quaternion_true,
        translation_true,
        quaternion_pred,
        translation_pred,
    ):
        quaternion_true = quaternion_true.astype(np.float32)
        translation_true = translation_true.astype(np.float32)

        B = class_id.shape[0]

        T_cad2cam_true = objslampp.functions.transformation_matrix(
            quaternion_true, translation_true
        )
        T_cad2cam_pred = objslampp.functions.transformation_matrix(
            quaternion_pred, translation_pred
        )
        T_cad2cam_true = cuda.to_cpu(T_cad2cam_true.array)
        T_cad2cam_pred = cuda.to_cpu(T_cad2cam_pred.array)

        summary = chainer.DictSummary()
        for i in range(B):
            class_id_i = int(class_id[i])
            cad_pcd = self._models.get_pcd(class_id=class_id_i)
            for translate in [True, False]:
                add, add_s = objslampp.metrics.average_distance(
                    points=[cad_pcd],
                    transform1=[T_cad2cam_true[i]],
                    transform2=[T_cad2cam_pred[i]],
                    translate=translate,
                )
                add, add_s = add[0], add_s[0]
                add_type = 'add' if translate else 'addr'
                if chainer.config.train:
                    summary.add({f'{add_type}': add, f'{add_type}_s': add_s})
                else:
                    summary.add({
                        f'{add_type}/{class_id_i:04d}/{i:04d}': add,
                        f'{add_type}_s/{class_id_i:04d}/{i:04d}': add_s,
                    })
        chainer.report(summary.compute_mean(), self)

    def loss(
        self,
        *,
        class_id,
        quaternion_true,
        translation_true,
        quaternion_pred,
        translation_pred,
        confidence_pred,
    ):
        quaternion_true = quaternion_true.astype(np.float32)
        translation_true = translation_true.astype(np.float32)

        T_cad2cam_true = objslampp.functions.transformation_matrix(
            quaternion_true, translation_true
        )
        T_cad2cam_pred = objslampp.functions.transformation_matrix(
            quaternion_pred, translation_pred
        )

        B = class_id.shape[0]

        loss = 0
        for i in range(B):
            class_id_i = int(class_id[i])

            if self._loss in [
                'add',
                'add_s',
                'add/add_s',
                'add+add_s',
            ]:
                if self._loss in ['add+add_s']:
                    is_symmetric = None
                elif self._loss in ['add']:
                    is_symmetric = False
                elif self._loss in ['add_s']:
                    is_symmetric = True
                else:
                    assert self._loss in ['add/add_s']
                    is_symmetric = class_id_i in \
                        objslampp.datasets.ycb_video.class_ids_symmetric
                cad_pcd = self._models.get_pcd(class_id=class_id_i)
                cad_pcd = self.xp.asarray(cad_pcd, dtype=np.float32)

            if self._loss in [
                'add',
                'add_s',
                'add/add_s',
            ]:
                assert is_symmetric in [True, False]
                loss_i = objslampp.functions.average_distance_l1(
                    points=cad_pcd,
                    transform1=T_cad2cam_true[i][None],
                    transform2=T_cad2cam_pred[i][None],
                    symmetric=is_symmetric,
                )[0]
            elif self._loss in ['add+add_s']:
                kwargs = dict(
                    points=cad_pcd,
                    transform1=T_cad2cam_true[i][None],
                    transform2=T_cad2cam_pred[i][None],
                )
                loss_add_i = objslampp.functions.average_distance_l1(
                    **kwargs, symmetric=False
                )[0]
                loss_add_s_i = objslampp.functions.average_distance_l1(
                    **kwargs, symmetric=True
                )[0]
                loss_i = (
                    self._loss_scale['add+add_s'] * loss_add_i +
                    (1 - self._loss_scale['add+add_s']) * loss_add_s_i
                )
            else:
                raise ValueError(f'unsupported loss: {self._loss}')

            loss_i = F.mean(
                loss_i * confidence_pred[i] -
                self._lambda_confidence * F.log(confidence_pred[i])
            )
            loss += loss_i
        loss /= B

        values = {'loss': loss}
        chainer.report(values, observer=self)

        return loss


class VoxelFeatureExtractor(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # conv1: 32
            self.conv1_1 = L.Convolution3D(32, 32, 3, 1, pad=1)
            self.conv1_2 = L.Convolution3D(32, 32, 1, 1, pad=0)
            self.conv1_3 = L.Convolution3D(32, 32, 1, 1, pad=0)
            # conv2: 32
            self.conv2_1 = L.Convolution3D(32, 64, 3, 1, pad=1)
            self.conv2_2 = L.Convolution3D(64, 64, 1, 1, pad=0)
            self.conv2_3 = L.Convolution3D(64, 64, 1, 1, pad=0)
            # conv3: 32
            self.conv3_1 = L.Convolution3D(64, 128, 3, 1, pad=1)
            self.conv3_2 = L.Convolution3D(128, 128, 1, 1, pad=0)
            self.conv3_3 = L.Convolution3D(128, 128, 1, 1, pad=0)

    def __call__(self, h):
        # conv1
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.conv1_3(h))
        # conv2
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.relu(self.conv2_3(h))
        # conv3
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        return h
