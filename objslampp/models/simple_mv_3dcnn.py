import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.resnet import ResNet50
import numpy as np
import trimesh.transformations as tf

import objslampp


class SimpleMV3DCNNModel(chainer.Chain):

    def __init__(self):
        super(SimpleMV3DCNNModel, self).__init__()

        initialW = chainer.initializers.Normal(0.01)
        with self.init_scope():
            # MV
            self.res = ResNet50(arch='he', pretrained_model='imagenet')
            self.res.pick = ['res4']
            self.res.remove_unused()
            self.conv5 = L.Convolution2D(
                in_channels=1024,
                out_channels=16,
                ksize=1,
                initialW=initialW,
            )

            # voxelization_3d
            # -> (16, 32, 32, 32)
            self.voxel_dimensions = (32, 32, 32)
            self.voxel_channels = 16

            # 3DCNN
            self.conv6 = L.Convolution3D(
                in_channels=16,
                out_channels=16,
                ksize=8,
                stride=2,
                pad=2,
                initialW=initialW
            )  # 32x32x32 -> 16x16x16
            self.conv7 = L.Convolution3D(
                in_channels=16,
                out_channels=16,
                ksize=3,
                stride=2,
                pad=1,
                initialW=initialW
            )  # 16x16x16 -> 8x8x8

            # concat
            # (16, 8, 8, 8) + (16, 8, 8, 8) -> (32, 8, 8, 8)
            # 32 * 8 * 8 * 8 = 16384

            # FC
            self.fc8 = L.Linear(16384, 1024, initialW=initialW)
            self.fc_quaternion = L.Linear(1024, 4, initialW=initialW)
            self.fc_translation = L.Linear(1024, 3, initialW=initialW)

    def __call__(self, **kwargs):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        for k, v in kwargs.items():
            print(k, type(v), v.shape)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        batch_size = kwargs['class_id'].shape[0]
        assert batch_size == 1, 'single batch_size is only supported'

        # class_id = kwargs['class_id'][0].astype(np.int32)
        pitch = kwargs['pitch'][0].astype(np.float32)
        gt_pose = kwargs['gt_pose'][0].astype(np.float32)

        # ---------------------------------------------------------------------
        # cad
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        cad_origin = kwargs['cad_origin'][0].astype(np.float32)
        cad_rgbs = kwargs['cad_rgbs'][0].astype(np.float32)
        cad_pcds = kwargs['cad_pcds'][0].astype(np.float32)

        mean = self.xp.asarray(self.res.mean)
        cad_rgbs = cad_rgbs.transpose(0, 3, 1, 2)
        cad_rgbs = cad_rgbs - mean[None]

        # MV
        h, = self.res(cad_rgbs)
        print(f'h_res: {h.shape}')
        h = F.relu(self.conv5(h))
        print(f'h_conv5: {h.shape}')

        # Voxelization3D
        h = F.resize_images(h, cad_rgbs.shape[2:])
        h_vox = []
        for i in range(len(h)):
            isnan = self.xp.isnan(cad_pcds[i]).any(axis=2)
            h_i = h[i].transpose(1, 2, 0)
            h_i = objslampp.functions.voxelization_3d(
                values=h_i[~isnan, :],
                points=cad_pcds[i][~isnan, :],
                origin=cad_origin,
                pitch=pitch,
                dimensions=self.voxel_dimensions,
                channels=self.voxel_channels,
            )
            h_i = h_i.transpose(3, 0, 1, 2)  # XYZC -> CXYZ
            print(f'h_i: {h_i.shape}')
            h_vox.append(h_i[None])
            del h_i, isnan
        h = F.concat(h_vox, axis=0)
        del h_vox
        print(f'h_vox: {h.shape}')
        h = F.max(h, axis=0)[None]
        print(f'h_vox_fused: {h.shape}')

        # 3DCNN
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h_cad = h
        print(f'h_cad: {h.shape}')

        del cad_origin, cad_rgbs, cad_pcds

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # ---------------------------------------------------------------------
        # scan
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        scan_origin = kwargs['scan_origin'][0].astype(np.float32)
        scan_rgbs = kwargs['scan_rgbs'][0].astype(np.float32)
        scan_pcds = kwargs['scan_pcds'][0].astype(np.float32)
        scan_masks = kwargs['scan_masks'][0]
        del kwargs

        scan_rgbs = scan_rgbs.transpose(0, 3, 1, 2)
        scan_rgbs = scan_rgbs - mean[None]

        # MV
        h, = self.res(scan_rgbs)
        print(f'h_res: {h.shape}')
        h = F.relu(self.conv5(h))
        print(f'h_conv5: {h.shape}')

        # Voxelization3D
        h = F.resize_images(h, scan_rgbs.shape[2:])
        h_vox = []
        for i in range(len(h)):
            isnan = self.xp.isnan(scan_pcds[i]).any(axis=2)
            mask = (~isnan) & scan_masks[i]
            h_i = h[i].transpose(1, 2, 0)
            h_i = objslampp.functions.voxelization_3d(
                values=h_i[mask, :],
                points=scan_pcds[i][mask, :],
                origin=scan_origin,
                pitch=pitch,
                dimensions=self.voxel_dimensions,
                channels=self.voxel_channels,
            )
            h_i = h_i.transpose(3, 0, 1, 2)
            print(f'h_i: {h_i.shape}')
            h_vox.append(h_i[None])
            del h_i, isnan
        h = F.concat(h_vox, axis=0)
        del h_vox
        print(f'h_vox: {h.shape}')
        h = F.max(h, axis=0)[None]
        print(f'h_vox_fused: {h.shape}')

        # 3DCNN
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h_scan = h
        print(f'h_scan: {h.shape}')

        del scan_rgbs, scan_pcds, scan_masks

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        # ---------------------------------------------------------------------
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        h = F.concat([h_cad, h_scan], axis=1)
        del h_cad, h_scan
        print(f'h_concat: {h.shape}')

        h = F.relu(self.fc8(h))
        print(f'h_fc8: {h.shape}')

        quaternion = F.sigmoid(self.fc_quaternion(h))
        print(f'quaternion: {quaternion}')
        translation = F.sigmoid(self.fc_translation(h))
        print(f'translation: {translation}')
        del h

        gt_pose = chainer.cuda.to_cpu(gt_pose)
        gt_quaternion = self.xp.asarray(tf.quaternion_from_matrix(gt_pose))
        gt_translation = self.xp.asarray(tf.translation_from_matrix(gt_pose))
        voxel_dimensions = self.xp.asarray(self.voxel_dimensions)
        gt_translation = \
            (gt_translation - scan_origin) / pitch / voxel_dimensions
        print(f'gt_quaternion: {gt_quaternion}')
        print(f'gt_translation: {gt_translation}')

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # import ipdb; ipdb.set_trace()  # NOQA
        quit()
