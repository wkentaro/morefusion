import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.vgg import VGG16
import numpy as np
import pybullet  # NOQA

import objslampp


class BaselineModel(chainer.Chain):

    def __init__(
        self,
        *,
        freeze_until,
        voxelization='max',
        lambda_confidence=0.015,
        n_point=1000,
    ):
        super().__init__()

        self._freeze_until = freeze_until
        self._voxelization = voxelization
        self._n_point = n_point
        self._lambda_confidence = lambda_confidence

        with self.init_scope():
            self.extractor = VGG16(pretrained_model='imagenet')
            self.extractor.pick = ['conv4_3', 'conv3_3', 'conv2_2', 'conv1_2']
            self.extractor.remove_unused()
            self.conv5 = L.Convolution2D(
                in_channels=512,
                out_channels=32,
                ksize=1,
                initialW=chainer.initializers.Normal(0.01),
            )

            # voxelization_3d -> (16, 32, 32, 32)
            self._voxel_dim = 32

            self.conv6 = L.Convolution3D(
                in_channels=32,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=1,
            )  # 32x32x32 -> 32x32x32
            self.conv7 = L.Convolution3D(
                in_channels=32,
                out_channels=32,
                ksize=3,
                stride=1,
                pad=1,
            )  # 32x32x32 -> 32x32x32

            # 16 * 8 * 8 * 8 = 8192
            self.fc8 = L.Linear(32, 32)
            self.fc_quaternion = L.Linear(32, 4)
            self.fc_translation = L.Linear(32, 3)
            self.fc_confidence = L.Linear(32, 1)

    def predict(
        self,
        *,
        class_id,
        pitch,
        rgb,
        pcd,
    ):
        xp = self.xp

        B, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3

        # prepare
        pitch = pitch.astype(np.float32)
        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # BHWC -> BCHW
        pcd = pcd.transpose(0, 3, 1, 2).astype(np.float32)  # BHW3 -> B3HW

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
        h = F.relu(self.conv5(h))
        h = F.resize_images(h, (H, W))

        h_vox = []
        origins = []
        for i in range(B):
            h_i = h[i]
            pcd_i = pcd[i]

            h_i = h_i.transpose(1, 2, 0)      # CHW -> HWC
            pcd_i = pcd_i.transpose(1, 2, 0)  # 3HW -> HW3
            mask_i = ~xp.isnan(pcd_i).any(axis=2)

            values = h_i[mask_i, :]    # MC
            points = pcd_i[mask_i, :]  # M3
            centroid = points.mean(axis=0)  # 3
            origin = centroid - pitch[i] * self._voxel_dim / 2.
            origins.append(origin)

            if self._voxelization == 'average':
                func = objslampp.functions.average_voxelization_3d
            else:
                assert self._voxelization == 'max'
                func = objslampp.functions.max_voxelization_3d
            h_i = func(
                values=values,
                points=points,
                origin=origin,
                pitch=pitch[i],
                dimensions=(self._voxel_dim,) * 3,
                channels=h_i.shape[2],
            )  # CXYZ
            h_vox.append(h_i[None])
        origins = xp.stack(origins, axis=0)  # B3
        h = F.concat(h_vox, axis=0)          # BCXYZ
        del h_vox

        nonzero = (h.array != 0).any(axis=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))

        h_point = []
        point_ijk = []
        for b in range(nonzero.shape[0]):
            n_point = int(nonzero[b].sum())
            if n_point >= self._n_point:
                keep = xp.random.permutation(n_point)[:self._n_point]
            else:
                keep = xp.r_[
                    xp.arange(n_point),
                    xp.random.randint(0, n_point, self._n_point - n_point),
                ]
            i, j, k = xp.where(nonzero[b])
            h_point.append(h[b:b + 1, :, i[keep], j[keep], k[keep]])
            point_ijk.append(xp.c_[i[keep], j[keep], k[keep]])
        h = F.concat(h_point, axis=0)
        del h_point
        point_ijk = xp.stack(point_ijk, axis=0)

        h = h.transpose(0, 2, 1)  # B,C,P -> B,P,C
        h = h.reshape(B * h.shape[1], h.shape[2])  # B,P,C -> B*P,C
        h = F.relu(self.fc8(h))

        quaternion_pred = F.normalize(self.fc_quaternion(h))
        quaternion_pred = quaternion_pred.reshape(B, self._n_point, 4)
        translation_pred = F.cos(self.fc_translation(h))
        translation_pred = translation_pred.reshape(B, self._n_point, 3)
        confidence_pred = F.sigmoid(self.fc_confidence(h))
        confidence_pred = confidence_pred.reshape(B, self._n_point, 1)

        # TODO(wkentaro): extend to multi-class
        offset = origins[:, None, :]
        # offset = origins[:, None, :] + point_ijk * pitch[:, None, None]
        translation_pred = (
            offset + translation_pred * self._voxel_dim * pitch[:, None, None]
        )
        confidence_pred = confidence_pred[:, :, 0]

        return quaternion_pred, translation_pred, confidence_pred

    def __call__(
        self,
        *,
        class_id,
        pitch,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        keep = class_id != -1
        if keep.sum() == 0:
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        class_id = class_id[keep]
        pitch = pitch[keep]
        rgb = rgb[keep]
        pcd = pcd[keep]
        quaternion_true = quaternion_true[keep]
        translation_true = translation_true[keep]

        quaternion_pred, translation_pred, confidence_pred = self.predict(
            class_id=class_id,
            pitch=pitch,
            rgb=rgb,
            pcd=pcd,
        )

        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
            confidence_pred=confidence_pred,
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
        confidence_pred,
    ):
        xp = self.xp
        B = class_id.shape[0]

        indices = F.argmax(confidence_pred, axis=1).array
        quaternion_pred = quaternion_pred[xp.arange(B), indices]
        translation_pred = translation_pred[xp.arange(B), indices]

        T_cad2cam_true = objslampp.functions.quaternion_matrix(quaternion_true)
        T_cad2cam_pred = objslampp.functions.quaternion_matrix(quaternion_pred)

        T_cad2cam_true = cuda.to_cpu(T_cad2cam_true.array)
        T_cad2cam_pred = cuda.to_cpu(T_cad2cam_pred.array)
        translation_true = cuda.to_cpu(translation_true)
        translation_pred = cuda.to_cpu(translation_pred.array)

        # add_rotation
        summary = chainer.DictSummary()
        for i in range(B):
            class_id_i = int(class_id[i])
            cad_pcd = self._get_cad_pcd(class_id=class_id_i)
            add_rotation = objslampp.metrics.average_distance(
                [cad_pcd], [T_cad2cam_true[i]], [T_cad2cam_pred[i]]
            )[0]
            if chainer.config.train:
                summary.add({'add_rotation': add_rotation})
            else:
                summary.add({f'add_rotation/{class_id_i:04d}': add_rotation})
        chainer.report(summary.compute_mean(), self)

        T_cad2cam_true = objslampp.functions.compose_transform(
            Rs=T_cad2cam_true[:, :3, :3], ts=translation_true,
        ).array
        T_cad2cam_pred = objslampp.functions.compose_transform(
            Rs=T_cad2cam_pred[:, :3, :3], ts=translation_pred
        ).array

        # add
        summary = chainer.DictSummary()
        for i in range(B):
            class_id_i = int(class_id[i])
            cad_pcd = self._get_cad_pcd(class_id=class_id_i)
            add = objslampp.metrics.average_distance(
                [cad_pcd], [T_cad2cam_true[i]], [T_cad2cam_pred[i]]
            )[0]
            if chainer.config.train:
                summary.add({'add': add})
            else:
                summary.add({f'add/{class_id_i:04d}': add})
        chainer.report(summary.compute_mean(), self)

    def loss(
        self,
        class_id,
        quaternion_true,
        translation_true,
        quaternion_pred,
        translation_pred,
        confidence_pred,
    ):
        xp = self.xp
        B = class_id.shape[0]

        # prepare
        quaternion_true = quaternion_true.astype(np.float32)
        translation_true = translation_true.astype(np.float32)

        loss = 0
        for i in range(B):
            n_point = quaternion_pred[i].shape[0]

            T_cad2cam_pred = objslampp.functions.quaternion_matrix(
                quaternion_pred[i]
            )
            T_cad2cam_pred = objslampp.functions.compose_transform(
                T_cad2cam_pred[:, :3, :3], translation_pred[i],
            )  # (M, 4, 4)

            T_cad2cam_true = objslampp.functions.quaternion_matrix(
                quaternion_true[i:i + 1]
            )
            T_cad2cam_true = objslampp.functions.compose_transform(
                T_cad2cam_true[:, :3, :3], translation_true[i:i + 1],
            )  # (1, 4, 4)
            T_cad2cam_true = F.repeat(T_cad2cam_true, n_point, axis=0)

            class_id_i = int(class_id[i])
            cad_pcd = self._get_cad_pcd(class_id=class_id_i)
            cad_pcd = xp.asarray(cad_pcd, dtype=np.float32)
            add = objslampp.functions.average_distance_l1(
                cad_pcd, T_cad2cam_true, T_cad2cam_pred
            )

            # loss_i = F.mean(add)
            loss_i = F.mean(
                add * confidence_pred[i] -
                self._lambda_confidence * F.log(confidence_pred[i])
            )
            loss += loss_i
        loss /= B

        values = {'loss': loss}
        chainer.report(values, observer=self)

        return loss

    def _get_cad_pcd(self, *, class_id):
        models = objslampp.datasets.YCBVideoModels()
        pcd_file = models.get_model(class_id=class_id)['points_xyz']
        return np.loadtxt(pcd_file)
