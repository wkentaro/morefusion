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
    _lambda_confidence = 0.015

    def __init__(
        self,
        *,
        n_fg_class,
    ):
        super().__init__()

        self._n_fg_class = n_fg_class

        with self.init_scope():
            # extractor
            self.resnet_extractor = objslampp.models.ResNet18()
            self.pspnet_extractor = PSPNetExtractor()
            self.voxel_extractor = VoxelFeatureExtractor()

            # fc1
            self.fc1_rot = L.Linear(2016, 640)
            self.fc1_trans = L.Linear(2016, 640)
            self.fc1_conf = L.Linear(2016, 640)
            # fc2
            self.fc2_rot = L.Linear(640, 256)
            self.fc2_trans = L.Linear(640, 256)
            self.fc2_conf = L.Linear(640, 256)
            # fc3
            self.fc3_rot = L.Linear(256, 128)
            self.fc3_trans = L.Linear(256, 128)
            self.fc3_conf = L.Linear(256, 128)
            # fc4
            self.fc4_rot = L.Linear(128, n_fg_class * 4)
            self.fc4_trans = L.Linear(128, n_fg_class * 3)
            self.fc4_conf = L.Linear(128, n_fg_class)

    def predict(
        self,
        *,
        class_id,
        rgb,
        pcd,
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

        batch_indices, quaternion_pred, translation_pred, confidence_pred = \
            self._predict_from_voxelized(
                class_id=class_id,
                pitch=pitch,
                origin=origin,
                voxelized=voxelized,
                count=count,
            )

        return (
            batch_indices, quaternion_pred, translation_pred, confidence_pred
        )

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

        batch_indices, values, points = self.voxel_extractor(voxelized, count)

        class_id = class_id[batch_indices]
        pitch = pitch[batch_indices]
        origin = origin[batch_indices]
        points = origin + points * pitch[:, None]

        h_rot = F.relu(self.fc1_rot(values))
        h_trans = F.relu(self.fc1_trans(values))
        h_conf = F.relu(self.fc1_conf(values))
        h_rot = F.relu(self.fc2_rot(h_rot))
        h_trans = F.relu(self.fc2_trans(h_trans))
        h_conf = F.relu(self.fc2_conf(h_conf))
        h_rot = F.relu(self.fc3_rot(h_rot))
        h_trans = F.relu(self.fc3_trans(h_trans))
        h_conf = F.relu(self.fc3_conf(h_conf))
        cls_rot = self.fc4_rot(h_rot)
        cls_trans = self.fc4_trans(h_trans)
        cls_conf = F.sigmoid(self.fc4_conf(h_conf))

        P = batch_indices.shape[0]

        quaternion = cls_rot.reshape(P, self._n_fg_class, 4)
        translation = cls_trans.reshape(P, self._n_fg_class, 3)
        confidence = cls_conf.reshape(P, self._n_fg_class)

        fg_class_id = class_id - 1
        quaternion = quaternion[xp.arange(P), fg_class_id, :]
        translation = translation[xp.arange(P), fg_class_id, :]
        confidence = confidence[xp.arange(P), fg_class_id]

        quaternion = F.normalize(quaternion, axis=1)
        translation = points + translation * pitch[:, None]

        return batch_indices, quaternion, translation, confidence

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

        batch_indices, quaternion_pred, translation_pred, confidence_pred = \
            self.predict(
                class_id=class_id,
                rgb=rgb,
                pcd=pcd,
            )

        quaternion_pred_selected = []
        translation_pred_selected = []
        for i in range(B):
            mask = batch_indices == i
            index = F.argmax(confidence_pred[mask]).array
            quaternion_pred_selected.append(quaternion_pred[mask][index])
            translation_pred_selected.append(translation_pred[mask][index])
        quaternion_pred_selected = F.stack(quaternion_pred_selected)
        translation_pred_selected = F.stack(translation_pred_selected)

        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred_selected,
            translation_pred=translation_pred_selected,
        )

        loss = self.loss(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            batch_indices=batch_indices,
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
        class_id,
        quaternion_true,
        translation_true,
        batch_indices,
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
            mask = batch_indices == i
            P = int(mask.sum())

            T_cad2cam_pred = objslampp.functions.transformation_matrix(
                quaternion_pred[mask], translation_pred[mask]
            )  # (M, 4, 4)

            T_cad2cam_true = objslampp.functions.transformation_matrix(
                quaternion_true[i], translation_true[i]
            )  # (4, 4)
            T_cad2cam_true = F.repeat(T_cad2cam_true[None], P, axis=0)

            class_id_i = int(class_id[i])
            is_symmetric = class_id_i in \
                objslampp.datasets.ycb_video.class_ids_symmetric
            cad_pcd = self._models.get_pcd(class_id=class_id_i)
            cad_pcd = xp.asarray(cad_pcd, dtype=np.float32)
            add = objslampp.functions.average_distance_l1(
                cad_pcd,
                T_cad2cam_true,
                T_cad2cam_pred,
                symmetric=is_symmetric,
            )

            loss_i = F.mean(
                add * confidence_pred[i] -
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
            self.conv1_1 = L.Convolution3D(32, 64, 4, 2, pad=1)
            self.conv1_2 = L.Convolution3D(64, 64, 1, 1, pad=0)
            # conv2: 16 -> 8
            self.conv2_1 = L.Convolution3D(64, 128, 4, 2, pad=1)
            self.conv2_2 = L.Convolution3D(128, 128, 1, 1, pad=0)
            # conv3: 8 -> 4
            self.conv3_1 = L.Convolution3D(128, 256, 4, 2, pad=1)
            self.conv3_2 = L.Convolution3D(256, 256, 1, 1, pad=0)
            # conv4: 4 -> 1
            self.conv4_1 = L.Convolution3D(256, 512, 4, 1, pad=0)
            self.conv4_2 = L.Convolution3D(512, 512, 1, 1, pad=0)
            # conv5: 1
            self.conv5 = L.Convolution3D(512, 1024, 1, 1, pad=0)

    def __call__(self, h, count):
        B = h.shape[0]
        xp = self.xp

        h_conv0 = h  # 32
        # conv1
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h_conv1 = h  # 16
        # conv2
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h_conv2 = h  # 8
        # conv3
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h_conv3 = h  # 4
        # conv4
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h_conv4 = h  # 2
        # conv5
        h = F.relu(self.conv5(h))
        h_conv5 = h  # 1

        batch_indices = []
        values = []
        points = []
        for i in range(B):
            I, J, K = xp.nonzero(count[i])
            P = len(I)
            if P > 1000:  # max points
                keep = xp.random.permutation(P)[:1000]
                I, J, K = I[keep], J[keep], K[keep]
                P = 1000
            h_conv0_i = h_conv0[i, :, I, J, K]
            h_conv1_i = h_conv1[i, :, I // 2, J // 2, K // 2]
            h_conv2_i = h_conv2[i, :, I // 4, J // 4, K // 4]
            h_conv3_i = h_conv3[i, :, I // 8, J // 8, K // 8]
            h_conv4_i = h_conv4[i, :, I // 32, J // 32, K // 32]
            h_conv5_i = h_conv5[i, :, I // 32, J // 32, K // 32]
            h_i = F.concat([
                h_conv0_i,
                h_conv1_i,
                h_conv2_i,
                h_conv3_i,
                h_conv4_i,
                h_conv5_i,
            ], axis=1)
            batch_indices.append(xp.full((P,), i, dtype=np.int32))
            values.append(h_i)
            points.append(xp.column_stack((I, J, K)))
        batch_indices = xp.concatenate(batch_indices, axis=0)
        values = F.concat(values, axis=0)
        points = xp.concatenate(points, axis=0)

        return batch_indices, values, points
