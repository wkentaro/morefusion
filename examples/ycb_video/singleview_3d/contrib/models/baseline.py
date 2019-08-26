import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import trimesh.transformations as tf

import objslampp

from .pspnet import PSPNetExtractor


class BaselineModel(chainer.Chain):

    _models = objslampp.datasets.YCBVideoModels()
    _voxel_dim = 32

    def __init__(
        self,
        *,
        n_fg_class,
        loss=None,
        loss_scale=None,
        randomize_base=False,
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

        self._randomize_base = randomize_base

        with self.init_scope():
            # extractor
            self.resnet_extractor = objslampp.models.ResNet18()
            self.pspnet_extractor = PSPNetExtractor()
            self.voxel_extractor = VoxelFeatureExtractor()

            self.fc1_rot = L.Linear(1024, 640, 1)
            self.fc1_trans = L.Linear(1024, 640, 1)
            self.fc2_rot = L.Linear(640, 256, 1)
            self.fc2_trans = L.Linear(640, 256, 1)
            self.fc3_rot = L.Linear(256, 128, 1)
            self.fc3_trans = L.Linear(256, 128, 1)
            self.fc4_rot = L.Linear(128, n_fg_class * 4, 1)
            self.fc4_trans = L.Linear(128, n_fg_class * 3, 1)

    def randomize_base(
        self, points, quaternion_true, translation_true
    ):
        xp = self.xp
        B = len(points)

        assert quaternion_true is not None
        assert translation_true is not None

        quaternion_true = quaternion_true.astype(np.float32)
        translation_true = translation_true.astype(np.float32)
        T_cad2cam_true = objslampp.functions.transformation_matrix(
            quaternion_true, translation_true
        ).array
        for i in range(B):
            T_cam2cad_true_i = F.inv(T_cad2cam_true[i]).array
            points[i] = objslampp.functions.transform_points(
                points[i], T_cam2cad_true_i
            ).array  # cad frame
            T_random_rot = xp.asarray(
                tf.random_rotation_matrix(), dtype=np.float32
            )
            T_cad2cam_true_i = T_cad2cam_true[i] @ T_random_rot
            points[i] = objslampp.functions.transform_points(
                points[i], T_cad2cam_true_i
            ).array  # cam frame
            quaternion_true[i] = xp.asarray(tf.quaternion_from_matrix(
                cuda.to_cpu(T_cad2cam_true_i)
            ), dtype=np.float32)

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

        if self._randomize_base and chainer.config.train:
            self.randomize_base(points, quaternion_true, translation_true)

        pitch, origin, voxelized, count = self._voxelize(
            class_id=class_id,
            values=values,
            points=points,
        )

        return self._predict_from_voxelized(
            class_id=class_id,
            pitch=pitch,
            origin=origin,
            voxelized=voxelized,
            count=count,
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

        h = self.voxel_extractor(voxelized, count)

        h_rot = F.relu(self.fc1_rot(h))
        h_trans = F.relu(self.fc1_trans(h))
        h_rot = F.relu(self.fc2_rot(h_rot))
        h_trans = F.relu(self.fc2_trans(h_trans))
        h_rot = F.relu(self.fc3_rot(h_rot))
        h_trans = F.relu(self.fc3_trans(h_trans))
        cls_rot = self.fc4_rot(h_rot)
        cls_trans = self.fc4_trans(h_trans)

        quaternion = cls_rot.reshape(B, self._n_fg_class, 4)
        translation = cls_trans.reshape(B, self._n_fg_class, 3)

        fg_class_id = class_id - 1
        quaternion = quaternion[xp.arange(B), fg_class_id, :]
        translation = translation[xp.arange(B), fg_class_id, :]

        quaternion = F.normalize(quaternion, axis=1)
        translation = origin + translation * pitch[:, None]

        return quaternion, translation

    def __call__(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        quaternion_pred, translation_pred = self.predict(
            class_id=class_id,
            rgb=rgb,
            pcd=pcd,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )

        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
        )

        loss = self.loss(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
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
            add, add_s = objslampp.metrics.average_distance(
                [cad_pcd], [T_cad2cam_true[i]], [T_cad2cam_pred[i]]
            )
            add, add_s = add[0], add_s[0]
            if chainer.config.train:
                summary.add({'add': add, 'add_s': add_s})
            else:
                summary.add({
                    f'add/{class_id_i:04d}/{i:04d}': add,
                    f'add_s/{class_id_i:04d}/{i:04d}': add_s,
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

            loss += loss_i
        loss /= B

        values = {'loss': loss}
        chainer.report(values, observer=self)

        return loss


class VoxelFeatureExtractor(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # conv1: 32 -> 16
            self.conv1_1_rgb = L.Convolution3D(32, 64, 4, stride=2, pad=1)
            self.conv1_2_rgb = L.Convolution3D(64, 64, 1, stride=1, pad=0)
            self.conv1_3_rgb = L.Convolution3D(64, 64, 1, stride=1, pad=0)
            self.conv1_1_ind = L.Convolution3D(3, 64, 4, stride=2, pad=1)
            self.conv1_2_ind = L.Convolution3D(64, 64, 1, stride=1, pad=0)
            self.conv1_3_ind = L.Convolution3D(64, 64, 1, stride=1, pad=0)
            # conv2: 16 -> 8
            self.conv2_1_rgb = L.Convolution3D(64, 128, 4, stride=2, pad=1)
            self.conv2_2_rgb = L.Convolution3D(128, 128, 1, stride=1, pad=0)
            self.conv2_3_rgb = L.Convolution3D(128, 128, 1, stride=1, pad=0)
            self.conv2_1_ind = L.Convolution3D(64, 128, 4, stride=2, pad=1)
            self.conv2_2_ind = L.Convolution3D(128, 128, 1, stride=1, pad=0)
            self.conv2_3_ind = L.Convolution3D(128, 128, 1, stride=1, pad=0)
            # conv3: 8
            self.conv3_1 = L.Convolution3D(256, 512, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution3D(512, 512, 1, stride=1, pad=0)
            self.conv3_3 = L.Convolution3D(512, 512, 1, stride=1, pad=0)
            # conv4: 8
            self.conv4_1 = L.Convolution3D(512, 1024, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution3D(1024, 1024, 1, stride=1, pad=0)
            self.conv4_3 = L.Convolution3D(1024, 1024, 1, stride=1, pad=0)

    def __call__(self, h_rgb, count):
        xp = self.xp
        B, _, X, Y, Z = h_rgb.shape

        h_ind = xp.stack(xp.meshgrid(xp.arange(X), xp.arange(Y), xp.arange(Z)))
        h_ind = xp.repeat(h_ind[None], B, axis=0)
        Ib, Ix, Iy, Iz = xp.where(count == 0)
        h_ind[Ib, :, Ix, Iy, Iz] = -1
        h_ind = h_ind.astype(np.float32)

        # conv1
        h_rgb = F.relu(self.conv1_1_rgb(h_rgb))
        h_rgb = F.relu(self.conv1_2_rgb(h_rgb))
        h_rgb = F.relu(self.conv1_3_rgb(h_rgb))
        h_ind = F.relu(self.conv1_1_ind(h_ind))
        h_ind = F.relu(self.conv1_2_ind(h_ind))
        h_ind = F.relu(self.conv1_3_ind(h_ind))
        # conv2
        h_rgb = F.relu(self.conv2_1_rgb(h_rgb))
        h_rgb = F.relu(self.conv2_2_rgb(h_rgb))
        h_rgb = F.relu(self.conv2_3_rgb(h_rgb))
        h_ind = F.relu(self.conv2_1_ind(h_ind))
        h_ind = F.relu(self.conv2_2_ind(h_ind))
        h_ind = F.relu(self.conv2_3_ind(h_ind))

        h = F.concat((h_rgb, h_ind), axis=1)

        # conv3
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        # conv4
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))

        h_avg = []
        for i in range(B):
            Ix, Iy, Iz = xp.where(count[i] > 0)
            Ix, Iy, Iz = Ix // 4, Iy // 4, Iz // 4  # 32 -> 8
            h_avg.append(F.average(h[i, :, Ix, Iy, Iz], axis=0))
        h_avg = F.stack(h_avg)
        return h_avg
