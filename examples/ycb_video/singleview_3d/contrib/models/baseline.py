import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import trimesh

import objslampp

from .pspnet import PSPNetExtractor


class BaselineModel(chainer.Chain):

    _models = objslampp.datasets.YCBVideoModels()
    _voxel_dim = 32

    def __init__(
        self,
        *,
        n_fg_class,
        use_occupancy=False,
        loss=None,
        loss_scale=None,
        ohem_threshold=None,
    ):
        super().__init__()

        self._n_fg_class = n_fg_class
        self._use_occupancy = use_occupancy

        self._loss = 'add/add_s' if loss is None else loss
        assert self._loss in [
            'add',
            'add+occupancy',
            'add_s',
            'add_s+occupancy',
            'add/add_s',
            'add/add_s+occupancy',
            'add+add_s',
            'overlap',
            'overlap+occupancy',
            'iou',
            'iou+occupancy',
        ]
        if self._loss in [
            'add+occupancy',
            'add_s+occupancy',
            'add/add_s+occupancy',
            'overlap+occupancy',
            'iou+occupancy',
        ]:
            assert use_occupancy, \
                f'use_occupancy must be True for this loss: {self._loss}'

        if loss_scale is None:
            loss_scale = {
                'add+add_s': 0.5,
                'occupancy': 1.0,
            }
        self._loss_scale = loss_scale

        if ohem_threshold is None:
            ohem_threshold = (0, 0)  # asymmetric, symmetric
        self._ohem_threshold = ohem_threshold

        with self.init_scope():
            # extractor
            self.resnet_extractor = objslampp.models.ResNet18()
            self.pspnet_extractor = PSPNetExtractor()

            if self._use_occupancy:
                self.conv5_occ = L.Convolution3D(
                    in_channels=1,
                    out_channels=8,
                    ksize=3,
                    stride=1,
                    pad=1,
                )  # 32x32x32 -> 32x32x32

            self.voxel_extractor = VoxelFeatureExtractor()

            self.fc1_rot = L.Linear(1024, 640, 1)
            self.fc1_trans = L.Linear(1024, 640, 1)
            self.fc2_rot = L.Linear(640, 256, 1)
            self.fc2_trans = L.Linear(640, 256, 1)
            self.fc3_rot = L.Linear(256, 128, 1)
            self.fc3_trans = L.Linear(256, 128, 1)
            self.fc4_rot = L.Linear(128, n_fg_class * 4, 1)
            self.fc4_trans = L.Linear(128, n_fg_class * 3, 1)

    def predict(
        self,
        *,
        class_id,
        pitch,
        origin,
        rgb,
        pcd,
        grid_nontarget_empty=None,
    ):
        xp = self.xp

        B, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3
        dimensions = (self._voxel_dim,) * 3

        # prepare
        pitch = pitch.astype(np.float32)
        origin = origin.astype(np.float32)
        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # BHWC -> BCHW
        pcd = pcd.transpose(0, 3, 1, 2).astype(np.float32)  # BHW3 -> B3HW
        if self._use_occupancy:
            assert grid_nontarget_empty.shape[1:] == dimensions
            grid_nontarget_empty = grid_nontarget_empty.astype(np.float32)

        # feature extraction
        mean = xp.asarray(self.resnet_extractor.mean)
        h = self.resnet_extractor(rgb - mean[None])  # 1/8
        h = self.pspnet_extractor(h)  # 1/1

        h_ = []
        counts = []
        centroids = []
        for i in range(B):
            h_i = h[i]
            pcd_i = pcd[i]

            h_i = h_i.transpose(1, 2, 0)      # CHW -> HWC
            pcd_i = pcd_i.transpose(1, 2, 0)  # 3HW -> HW3
            mask_i = ~xp.isnan(pcd_i).any(axis=2)

            values = h_i[mask_i, :]    # MC
            points = pcd_i[mask_i, :]  # M3

            h_i, counts_i = objslampp.functions.average_voxelization_3d(
                values=values,
                points=points,
                origin=origin[i],
                pitch=pitch[i],
                dimensions=dimensions,
                channels=h_i.shape[2],
                return_counts=True,
            )  # CXYZ
            h_.append(h_i[None])
            counts.append(counts_i[None])

            # mean of active points (voxels)
            centroid = xp.stack(xp.nonzero(counts_i[0])).mean(axis=1)
            centroid = centroid * pitch[i] + origin[i]
            centroids.append(centroid[None])
        h = F.concat(h_, axis=0)           # BCXYZ
        counts = xp.concatenate(counts, axis=0)  # BXYZ
        centroids = xp.concatenate(centroids, axis=0)  # B3
        del h_

        if self._use_occupancy:
            h_occ = self.conv5_occ(grid_nontarget_empty[:, None, :, :, :])
            h = F.concat([h, h_occ], axis=1)

        h = self.voxel_extractor(h, counts)

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
        translation = centroids + translation * pitch[:, None]

        return quaternion, translation

    def __call__(
        self,
        *,
        class_id,
        pitch,
        origin,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
        grid_target=None,
        grid_nontarget_empty=None,
    ):
        keep = class_id != -1
        if keep.sum() == 0:
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        class_id = class_id[keep]
        pitch = pitch[keep]
        origin = origin[keep]
        rgb = rgb[keep]
        pcd = pcd[keep]
        quaternion_true = quaternion_true[keep]
        translation_true = translation_true[keep]
        if self._use_occupancy:
            assert grid_target is not None
            assert grid_nontarget_empty is not None
            grid_target = grid_target[keep]
            grid_nontarget_empty = grid_nontarget_empty[keep]

        quaternion_pred, translation_pred = self.predict(
            class_id=class_id,
            pitch=pitch,
            origin=origin,
            rgb=rgb,
            pcd=pcd,
            grid_nontarget_empty=grid_nontarget_empty,
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
            pitch=pitch,
            origin=origin,
            grid_target=grid_target,
            grid_nontarget_empty=grid_nontarget_empty,
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

        batch_size = class_id.shape[0]

        T_cad2cam_true = objslampp.functions.quaternion_matrix(quaternion_true)
        T_cad2cam_pred = objslampp.functions.quaternion_matrix(quaternion_pred)
        T_cad2cam_true = objslampp.functions.compose_transform(
            Rs=T_cad2cam_true[:, :3, :3], ts=translation_true,
        )
        T_cad2cam_pred = objslampp.functions.compose_transform(
            Rs=T_cad2cam_pred[:, :3, :3], ts=translation_pred,
        )
        T_cad2cam_true = cuda.to_cpu(T_cad2cam_true.array)
        T_cad2cam_pred = cuda.to_cpu(T_cad2cam_pred.array)

        summary = chainer.DictSummary()
        for i in range(batch_size):
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
                    f'add/{class_id_i:04d}': add,
                    f'add_s/{class_id_i:04d}': add_s,
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
        pitch=None,
        origin=None,
        grid_target=None,
        grid_nontarget_empty=None,
    ):
        quaternion_true = quaternion_true.astype(np.float32)
        translation_true = translation_true.astype(np.float32)
        pitch = None if pitch is None else pitch.astype(np.float32)
        origin = None if origin is None else origin.astype(np.float32)
        if grid_target is not None:
            grid_target = grid_target.astype(np.float32)
        if grid_nontarget_empty is not None:
            grid_nontarget_empty = grid_nontarget_empty.astype(np.float32)

        R_cad2cam_true = objslampp.functions.quaternion_matrix(quaternion_true)
        R_cad2cam_pred = objslampp.functions.quaternion_matrix(quaternion_pred)
        del quaternion_true
        del quaternion_pred

        T_cad2cam_true = objslampp.functions.compose_transform(
            R_cad2cam_true[:, :3, :3], translation_true,
        )
        T_cad2cam_pred1 = objslampp.functions.compose_transform(
            R_cad2cam_pred[:, :3, :3], translation_true,
        )
        T_cad2cam_pred2 = objslampp.functions.compose_transform(
            R_cad2cam_pred[:, :3, :3], translation_pred,
        )
        del translation_true
        del translation_pred
        del R_cad2cam_true
        del R_cad2cam_pred

        batch_size = class_id.shape[0]

        loss = 0
        for i in range(batch_size):
            class_id_i = int(class_id[i])

            if self._loss in [
                'add',
                'add+occupancy',
                'add_s',
                'add_s+occupancy',
                'add/add_s',
                'add/add_s+occupancy',
                'add+add_s',
            ]:
                if self._loss in ['add+add_s']:
                    is_symmetric = None
                elif self._loss in ['add', 'add+occupancy']:
                    is_symmetric = False
                elif self._loss in ['add_s', 'add_s+occupancy']:
                    is_symmetric = True
                else:
                    assert self._loss in ['add/add_s', 'add/add_s+occupancy']
                    is_symmetric = class_id_i in \
                        objslampp.datasets.ycb_video.class_ids_symmetric
                cad_pcd = self._models.get_pcd(class_id=class_id_i)
                cad_pcd = self.xp.asarray(cad_pcd, dtype=np.float32)

            if self._loss in [
                'add+occupancy',
                'add_s+occupancy',
                'add/add_s+occupancy',
                'overlap',
                'overlap+occupancy',
                'iou',
                'iou+occupancy',
            ]:
                solid_pcd = self._models.get_solid_voxel(class_id=class_id_i)
                solid_pcd = self.xp.asarray(solid_pcd.points, dtype=np.float32)
                kwargs = dict(
                    pitch=float(pitch[i]),
                    origin=cuda.to_cpu(origin[i]),
                    dims=(self._voxel_dim,) * 3,
                    threshold=2.0,
                )
                grid_target_pred1 = \
                    objslampp.functions.pseudo_occupancy_voxelization(
                        points=objslampp.functions.transform_points(
                            solid_pcd, T_cad2cam_pred1[i][None]
                        )[0],
                        **kwargs,
                    )
                grid_target_pred2 = \
                    objslampp.functions.pseudo_occupancy_voxelization(
                        points=objslampp.functions.transform_points(
                            solid_pcd, T_cad2cam_pred2[i][None]
                        )[0],
                        **kwargs,
                    )
                if self._loss in [
                    'overlap',
                    'overlap+occupancy',
                    'iou',
                    'iou+occupancy',
                ]:
                    pcd_true = objslampp.functions.transform_points(
                        solid_pcd, T_cad2cam_true[i][None]
                    )[0]
                    pcd_true = cuda.to_cpu(pcd_true.array)
                    indices = trimesh.voxel.points_to_indices(
                        pcd_true,
                        pitch=kwargs['pitch'],
                        origin=kwargs['origin'],
                    )
                    del pcd_true
                    grid_target_true = self.xp.zeros(
                        kwargs['dims'], dtype=np.float32
                    )
                    grid_target_true[
                        indices[:, 0], indices[:, 1], indices[:, 2]
                    ] = 1
                    del indices
                del solid_pcd

            if self._loss in [
                'add',
                'add+occupancy',
                'add_s',
                'add_s+occupancy',
                'add/add_s',
                'add/add_s+occupancy',
            ]:
                assert is_symmetric in [True, False]
                loss_i = objslampp.functions.average_distance_l1(
                    points=cad_pcd,
                    transform1=T_cad2cam_true[i][None],
                    transform2=T_cad2cam_pred2[i][None],
                    symmetric=is_symmetric,
                    ohem_threshold=self._ohem_threshold[is_symmetric],
                )[0]
            elif self._loss in ['add+add_s']:
                kwargs = dict(
                    points=cad_pcd,
                    transform1=T_cad2cam_true[i][None],
                    transform2=T_cad2cam_pred2[i][None],
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
            elif self._loss in ['overlap', 'overlap+occupancy']:
                intersection = F.sum(grid_target_pred2 * grid_target_true)
                denominator = F.sum(grid_target_true) + 1e-16
                loss_i = - intersection / denominator
            elif self._loss in ['iou', 'iou+occupancy']:
                intersection = grid_target_pred2 * grid_target_true
                union = grid_target_pred2 + grid_target_true - intersection
                loss_i = 1 - F.sum(intersection) / F.sum(union)
            else:
                raise ValueError(f'unsupported loss: {self._loss}')

            if self._loss in [
                'add+occupancy',
                'add_s+occupancy',
                'add/add_s+occupancy',
                'overlap+occupancy',
                'iou+occupancy',
            ]:
                intersection = F.sum(grid_target_pred2 * grid_target[i])
                denominator = F.sum(grid_target_pred2) + 1e-16
                loss_i += (
                    self._loss_scale['occupancy'] * intersection / denominator
                )
                intersection = F.sum(
                    grid_target_pred1 * grid_nontarget_empty[i]
                )
                denominator = F.sum(grid_target_pred1) + 1e-16
                loss_i += (
                    self._loss_scale['occupancy'] * intersection / denominator
                )
            loss += loss_i
        loss /= batch_size

        values = {'loss': loss}
        chainer.report(values, observer=self)

        return loss


class VoxelFeatureExtractor(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution3D(None, 128, 4, stride=2, pad=1)
            self.conv2 = L.Convolution3D(128, 256, 3, stride=1, pad=1)
            self.conv3 = L.Convolution3D(256, 512, 3, stride=1, pad=1)
            self.conv4 = L.Convolution3D(512, 1024, 3, stride=1, pad=1)

    def __call__(self, h, counts):
        xp = self.xp

        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h_ = []
        for i in range(h.shape[0]):
            X, Y, Z = xp.nonzero(counts[i, 0, :, :, :])
            X, Y, Z = X // 2, Y // 2, Z // 2
            h_i = h[i, :, X, Y, Z]
            h_i = F.average(h_i, axis=0)
            h_.append(h_i[None])
        h = F.concat(h_, axis=0)
        del h_

        return h
