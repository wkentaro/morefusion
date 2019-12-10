import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

import morefusion


class Model(chainer.Chain):

    _models = morefusion.datasets.YCBVideoModels()
    _voxel_dim = 32
    _lambda_confidence = 0.015
    _n_point = 1000

    def __init__(
        self,
        *,
        n_fg_class,
        pretrained_resnet18=False,
        with_count=False,
    ):
        super().__init__()

        self._n_fg_class = n_fg_class

        with self.init_scope():
            # extractor
            if pretrained_resnet18:
                self.resnet_extractor = morefusion.models.ResNet18Extractor()
            else:
                self.resnet_extractor = \
                    morefusion.models.dense_fusion.ResNet18()
            self.pspnet_extractor = \
                morefusion.models.dense_fusion.PSPNetExtractor()
            self.voxel_extractor = VoxelFeatureExtractor(
                self._n_point, with_count
            )

            # fc1
            self.conv1_rot = L.Convolution1D(992, 640, 1)
            self.conv1_trans = L.Convolution1D(992, 640, 1)
            self.conv1_conf = L.Convolution1D(992, 640, 1)
            # conv2
            self.conv2_rot = L.Convolution1D(640, 256, 1)
            self.conv2_trans = L.Convolution1D(640, 256, 1)
            self.conv2_conf = L.Convolution1D(640, 256, 1)
            # conv3
            self.conv3_rot = L.Convolution1D(256, 128, 1)
            self.conv3_trans = L.Convolution1D(256, 128, 1)
            self.conv3_conf = L.Convolution1D(256, 128, 1)
            # conv4
            self.conv4_rot = L.Convolution1D(128, n_fg_class * 4, 1)
            self.conv4_trans = L.Convolution1D(128, n_fg_class * 3, 1)
            self.conv4_conf = L.Convolution1D(128, n_fg_class, 1)

    def predict(
        self,
        *,
        class_id,
        rgb,
        pcd,
        pitch=None,
        origin=None,
        grid_nontarget_empty=None,
    ):
        values, points = self._extract(
            rgb=rgb,
            pcd=pcd,
        )

        pitch, origin, voxelized, count = self._voxelize(
            class_id=class_id,
            values=values,
            points=points,
            pitch=pitch,
            origin=origin,
            grid_nontarget_empty=grid_nontarget_empty,
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
        h_rgb = self.resnet_extractor(rgb)  # 1/8
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
        pitch=None,
        origin=None,
        grid_nontarget_empty=None,
    ):
        xp = self.xp

        B = class_id.shape[0]
        dimensions = (self._voxel_dim,) * 3

        if pitch is None:
            pitch = [None] * B
        if origin is None:
            origin = [None] * B

        voxelized = []
        count = []
        for i in range(B):
            if pitch[i] is None:
                pitch[i] = self._models.get_voxel_pitch(
                    dimension=self._voxel_dim, class_id=int(class_id[i]),
                )
            if origin[i] is None:
                center_i = morefusion.extra.cupy.median(points[i], axis=0)
                origin[i] = center_i - pitch[i] * (self._voxel_dim / 2. - 0.5)
            voxelized_i, count_i = morefusion.functions.average_voxelization_3d(
                values=values[i],
                points=points[i],
                origin=origin[i],
                pitch=pitch[i],
                dimensions=dimensions,
                return_counts=True,
            )
            voxelized.append(voxelized_i)
            count.append(count_i)
        pitch = xp.asarray(pitch, dtype=np.float32)
        origin = xp.stack(origin).astype(np.float32)
        voxelized = F.stack(voxelized)
        count = xp.stack(count)

        if grid_nontarget_empty is not None:
            grid_nontarget_empty = grid_nontarget_empty.astype(np.float32)
            voxelized = F.concat(
                [voxelized, grid_nontarget_empty[:, None, :, :, :]], axis=1
            )

        return pitch, origin, voxelized, count

    def _predict_from_voxelized(
        self, class_id, pitch, origin, voxelized, count
    ):
        B = class_id.shape[0]
        xp = self.xp

        values, points = self.voxel_extractor(voxelized, count)

        # prepare
        pitch = pitch.astype(np.float32)
        origin = origin.astype(np.float32)

        points = (
            origin[:, :, None] +
            points.astype(np.float32) *
            pitch[:, None, None]
        )

        h_rot = F.relu(self.conv1_rot(values))
        h_trans = F.relu(self.conv1_trans(values))
        h_conf = F.relu(self.conv1_conf(values))
        h_rot = F.relu(self.conv2_rot(h_rot))
        h_trans = F.relu(self.conv2_trans(h_trans))
        h_conf = F.relu(self.conv2_conf(h_conf))
        h_rot = F.relu(self.conv3_rot(h_rot))
        h_trans = F.relu(self.conv3_trans(h_trans))
        h_conf = F.relu(self.conv3_conf(h_conf))
        cls_rot = self.conv4_rot(h_rot)
        cls_trans = self.conv4_trans(h_trans)
        cls_conf = F.sigmoid(self.conv4_conf(h_conf))

        quaternion = cls_rot.reshape(B, self._n_fg_class, 4, self._n_point)
        translation = cls_trans.reshape(B, self._n_fg_class, 3, self._n_point)
        confidence = cls_conf.reshape(B, self._n_fg_class, self._n_point)

        fg_class_id = class_id - 1
        assert (fg_class_id >= 0).all()
        quaternion = quaternion[xp.arange(B), fg_class_id, :, :]
        translation = translation[xp.arange(B), fg_class_id, :, :]
        confidence = confidence[xp.arange(B), fg_class_id, :]

        quaternion = F.normalize(quaternion, axis=1)
        translation = points + translation * pitch[:, None, None]

        quaternion = quaternion.transpose(0, 2, 1)    # B4M -> BM4
        translation = translation.transpose(0, 2, 1)  # B3M -> BM3

        return quaternion, translation, confidence

    def __call__(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
        pitch=None,
        origin=None,
        grid_nontarget_empty=None,
    ):
        B = class_id.shape[0]
        xp = self.xp

        quaternion_pred, translation_pred, confidence_pred = self.predict(
            class_id=class_id,
            rgb=rgb,
            pcd=pcd,
            pitch=pitch,
            origin=origin,
            grid_nontarget_empty=grid_nontarget_empty,
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

        T_cad2cam_true = morefusion.functions.transformation_matrix(
            quaternion_true, translation_true
        )
        T_cad2cam_pred = morefusion.functions.transformation_matrix(
            quaternion_pred, translation_pred
        )
        T_cad2cam_true = cuda.to_cpu(T_cad2cam_true.array)
        T_cad2cam_pred = cuda.to_cpu(T_cad2cam_pred.array)

        summary = chainer.DictSummary()
        for i in range(B):
            class_id_i = int(class_id[i])
            cad_pcd = self._models.get_pcd(class_id=class_id_i)
            for translate in [True, False]:
                add, add_s = morefusion.metrics.average_distance(
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

            T_cad2cam_pred = morefusion.functions.transformation_matrix(
                quaternion_pred[i], translation_pred[i]
            )  # (M, 4, 4)

            T_cad2cam_true = morefusion.functions.transformation_matrix(
                quaternion_true[i], translation_true[i]
            )  # (4, 4)
            T_cad2cam_true = F.repeat(T_cad2cam_true[None], n_point, axis=0)

            class_id_i = int(class_id[i])
            is_symmetric = class_id_i in \
                morefusion.datasets.ycb_video.class_ids_symmetric
            cad_pcd = self._models.get_pcd(class_id=class_id_i)
            cad_pcd = xp.asarray(cad_pcd, dtype=np.float32)
            add = morefusion.functions.average_distance_l1(
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

    def __init__(self, n_point, with_count):
        self._n_point = n_point
        self._with_count = with_count
        super().__init__()
        with self.init_scope():
            C = [None, 32, 64, 128, 256, 512]
            self.conv1_1 = L.Convolution3D(C[0], C[1], 3, 1, pad=1)  # 32
            # self.conv1_2 = L.Convolution3D(C[1], C[1], 1, 1, pad=0)
            self.conv2_1 = L.Convolution3D(C[1], C[2], 4, 2, pad=1)  # 32 -> 16
            # self.conv2_2 = L.Convolution3D(C[2], C[2], 1, 1, pad=0)
            self.conv3_1 = L.Convolution3D(C[2], C[3], 4, 2, pad=1)  # 16 -> 8
            # self.conv3_2 = L.Convolution3D(C[3], C[3], 1, 1, pad=0)
            self.conv4_1 = L.Convolution3D(C[3], C[4], 4, 2, pad=1)  # 8 -> 4
            # self.conv4_2 = L.Convolution3D(C[4], C[4], 1, 1, pad=0)
            self.conv5_1 = L.Convolution3D(C[4], C[5], 4, 1, pad=0)  # 4 -> 1
            # self.conv5_2 = L.Convolution3D(C[5], C[5], 1, 1, pad=0)

    def __call__(self, h, count):
        B, _, X, Y, Z = h.shape
        xp = self.xp

        h_ind = xp.stack(xp.meshgrid(xp.arange(X), xp.arange(Y), xp.arange(Z)))
        h_ind = h_ind[None].repeat(B, axis=0)
        assert X == Y == Z == 32
        h_ind = (X / 2.0 - 0.5) - h_ind.astype(np.float32)
        assert h_ind.shape == (B, 3, X, Y, Z)

        h_count = count.astype(np.float32)[:, None, :, :, :]

        if self._with_count:
            h = F.concat([h, h_ind, h_count], axis=1)
        else:
            h = F.concat([h, h_ind], axis=1)

        # conv1
        h = F.relu(self.conv1_1(h))
        # h = F.relu(self.conv1_2(h))
        h_conv1 = h
        assert h_conv1.shape == (B, 32, 32, 32, 32)
        # conv2
        h = F.relu(self.conv2_1(h))
        # h = F.relu(self.conv2_2(h))
        h_conv2 = h
        assert h_conv2.shape == (B, 64, 16, 16, 16)
        # conv3
        h = F.relu(self.conv3_1(h))
        # h = F.relu(self.conv3_2(h))
        h_conv3 = h
        assert h_conv3.shape == (B, 128, 8, 8, 8)
        # conv4
        h = F.relu(self.conv4_1(h))
        # h = F.relu(self.conv4_2(h))
        h_conv4 = h
        assert h_conv4.shape == (B, 256, 4, 4, 4)
        # conv5
        h = F.relu(self.conv5_1(h))
        # h = F.relu(self.conv5_2(h))
        h_conv5 = h
        assert h_conv5.shape == (B, 512, 1, 1, 1)

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
                    xp.random.randint(0, n_point, self._n_point - n_point),
                ]
            assert keep.shape == (self._n_point,)
            I, J, K = I[keep], J[keep], K[keep]
            # indices = xp.column_stack((I, J, K)).astype(np.float32)
            values_i = F.concat([
                h_conv1[i, :, I, J, K],
                h_conv2[i, :, I // 2, J // 2, K // 2],
                h_conv3[i, :, I // 4, J // 4, K // 4],
                h_conv4[i, :, I // 8, J // 8, K // 8],
                # morefusion.functions.interpolate_voxel_grid(
                #     h_conv2[i], indices / 2.
                # ),
                # morefusion.functions.interpolate_voxel_grid(
                #     h_conv3[i], indices / 4.
                # ),
                # morefusion.functions.interpolate_voxel_grid(
                #     h_conv4[i], indices / 8.
                # ),
                F.repeat(h_conv5[i, :, 0, 0, 0][None], self._n_point, axis=0),
            ], axis=1)
            values.append(values_i)
            points.append(xp.column_stack((I, J, K)))
        values = F.stack(values)
        points = xp.stack(points)

        values = values.transpose(0, 2, 1)  # BMC -> BCM
        points = points.transpose(0, 2, 1)  # BM3 -> B3M

        return values, points
