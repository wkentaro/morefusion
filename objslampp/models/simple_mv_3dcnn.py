import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.resnet import ResNet50
from chainercv.links.model.vgg import VGG16
import numpy as np

from .. import geometry
from .. import functions
from ..logger import logger


class SimpleMV3DCNNModel(chainer.Chain):

    def __init__(
        self,
        extractor,
        lambda_quaternion=1.0,
        lambda_translation=1.0,
    ):
        super(SimpleMV3DCNNModel, self).__init__()

        self._lambda_quaternion = lambda_quaternion
        self._lambda_translation = lambda_translation

        initialW = chainer.initializers.Normal(0.01)
        with self.init_scope():
            # MV
            if extractor == 'resnet50':
                self.extractor = ResNet50(
                    arch='he', pretrained_model='imagenet'
                )
                self.extractor.pick = ['res4']
                in_channels = 1024
            else:
                assert extractor == 'vgg16'
                self.extractor = VGG16(pretrained_model='imagenet')
                self.extractor.pick = ['pool4']
                in_channels = 512
            self.extractor.remove_unused()

            self.conv5 = L.Convolution2D(
                in_channels=in_channels,
                out_channels=16,
                ksize=1,
                initialW=initialW,
            )

            # voxelization_3d
            # -> (16, 32, 32, 32)
            self.voxel_dim = 32
            self.voxel_channels = 16

            # 3DCNN
            self.conv6 = L.Convolution3D(
                in_channels=16,
                out_channels=16,
                ksize=8,
                stride=2,
                pad=3,
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

    def _encode_multiview(self, origin, pitch, rgbs, pcds, masks):
        batch_size = len(rgbs)
        assert batch_size == len(pcds)
        assert batch_size == len(masks)

        # MV
        mean = self.xp.asarray(self.extractor.mean)
        rgbs = rgbs - mean[None]
        if isinstance(self.extractor, ResNet50):
            with chainer.using_config('train', False):
                h, = self.extractor(rgbs)
        else:
            assert isinstance(self.extractor, VGG16)
            h, = self.extractor(rgbs)
        logger.debug(f'h_extractor: {h.shape}')

        h = F.relu(self.conv5(h))
        logger.debug(f'h_conv5: {h.shape}')

        masks = (~self.xp.isnan(pcds).any(axis=3)) & masks
        bboxes = geometry.masks_to_bboxes(chainer.cuda.to_cpu(masks))
        h = F.resize_images(h, rgbs.shape[2:4])

        h_vox = []
        for i in range(batch_size):
            h_i = h[i]
            pcd = pcds[i]
            mask = masks[i]
            bbox = bboxes[i]

            y1, x1, y2, x2 = bbox.round().astype(int).tolist()
            if ((y2 - y1) * (x2 - x1)) == 0:
                continue

            h_i = h_i[:, y1:y2, x1:x2]  # CHW
            mask = mask[y1:y2, x1:x2]   # HW
            pcd = pcd[y1:y2, x1:x2, :]  # HWC

            h_i = F.resize_images(h_i[None, :, :, :], (128, 128))[0]
            pcd = pcd.transpose(2, 0, 1)  # HWC -> CHW
            pcd = F.resize_images(pcd[None, :, :, :], (128, 128))[0]
            pcd = pcd.transpose(1, 2, 0).array  # CHW -> HWC
            mask = mask.astype(np.float32)
            mask = F.resize_images(mask[None, None, :, :], (128, 128))[0, 0]
            mask = mask.array > 0.5
            mask = (~self.xp.isnan(pcd).any(axis=2)) & mask

            h_i = h_i.transpose(1, 2, 0)  # CHW -> HWC
            h_i = functions.voxelization_3d(
                values=h_i[mask, :],
                points=pcd[mask, :],
                origin=origin,
                pitch=pitch,
                dimensions=(self.voxel_dim,) * 3,
                channels=self.voxel_channels,
            )
            h_i = h_i.transpose(3, 0, 1, 2)  # XYZC -> CXYZ
            logger.debug(f'h_i, i={i}: {h_i.shape}')
            h_vox.append(h_i[None])
        h = F.concat(h_vox, axis=0)
        logger.debug(f'h_vox: {h.shape}')
        h = F.max(h, axis=0)[None]
        logger.debug(f'h_vox_fused: {h.shape}')  # NOQA

        # 3DCNN
        h = F.relu(self.conv6(h))
        logger.debug(f'h_conv6: {h.shape}')
        h = F.relu(self.conv7(h))
        logger.debug(f'h_conv7: {h.shape}')
        return h

    def _predict_pose(self, h_cad, h_scan):
        h = F.concat([h_cad, h_scan], axis=1)
        logger.debug(f'h_concat: {h.shape}')

        h = F.relu(self.fc8(h))
        logger.debug(f'h_fc8: {h.shape}')

        quaternion = F.normalize(self.fc_quaternion(h))  # [-1, 1]
        logger.debug(f'quaternion: {quaternion}')
        translation = F.cos(self.fc_translation(h))      # [-1, 1]
        logger.debug(f'translation: {translation}')
        return quaternion, translation

    def predict(
        self,
        *,
        class_id,
        pitch,
        cad_origin,
        cad_rgbs,
        cad_pcds,
        scan_origin,
        scan_rgbs,
        scan_pcds,
        scan_masks,
    ):
        batch_size = class_id.shape[0]
        assert batch_size == 1
        pitch = pitch[0]
        class_id = class_id[0]

        assert class_id > 0  # 0 indicates background class

        logger.debug('==> Multi-View Encoding CAD')
        cad_origin = cad_origin[0]
        cad_rgbs = cad_rgbs[0].astype(np.float32).transpose(0, 3, 1, 2)
        cad_pcds = cad_pcds[0]
        cad_masks = ~self.xp.isnan(cad_pcds).any(axis=3)
        h_cad = self._encode_multiview(
            origin=cad_origin,
            pitch=pitch,
            rgbs=cad_rgbs,
            pcds=cad_pcds,
            masks=cad_masks,
        )

        logger.debug('==> Multi-View Encoding Scan')
        scan_origin = scan_origin[0]
        scan_rgbs = scan_rgbs[0].astype(np.float32).transpose(0, 3, 1, 2)
        scan_pcds = scan_pcds[0]
        scan_masks = scan_masks[0] & ~self.xp.isnan(scan_pcds).any(axis=3)
        h_scan = self._encode_multiview(
            origin=scan_origin,
            pitch=pitch,
            rgbs=scan_rgbs,
            pcds=scan_pcds,
            masks=scan_masks
        )

        logger.debug('==> Predicting Pose')
        quaternion, translation = self._predict_pose(h_cad, h_scan)

        return quaternion, translation

    def __call__(
        self,
        *,
        valid,
        video_id,
        class_id,
        pitch,
        cad_origin,
        cad_rgbs,
        cad_pcds,
        scan_origin,
        scan_rgbs,
        scan_pcds,
        scan_masks,
        gt_pose,
        gt_quaternion,
        gt_translation,
    ):
        logger.debug('==> Arguments for SimpleMV3DCNNModel')
        logger.debug(f'valid: {type(valid)}, {valid}')
        logger.debug(f'video_id: {type(video_id)}, {video_id}')
        logger.debug(f'class_id: {type(class_id)}, {class_id}')
        logger.debug(f'pitch: {type(pitch)}, {pitch}')
        logger.debug(f'cad_origin: {type(cad_origin)}, {cad_origin}')
        logger.debug(f'cad_rgbs: {type(cad_rgbs)}, {cad_rgbs.shape}')
        logger.debug(f'cad_pcds: {type(cad_pcds)}, {cad_pcds.shape}')
        logger.debug(f'scan_origin: {type(scan_origin)}, {scan_origin.shape}')
        logger.debug(f'scan_rgbs: {type(scan_rgbs)}, {scan_rgbs.shape}')
        logger.debug(f'scan_pcds: {type(scan_pcds)}, {scan_pcds.shape}')
        logger.debug(f'scan_masks: {type(scan_masks)}, {scan_masks.shape}')
        if gt_pose is not None:
            logger.debug(f'gt_pose: {type(gt_pose)}, {gt_pose.shape}')
        if gt_quaternion is not None:
            logger.debug(f'gt_quaternion: {type(gt_quaternion)}, {gt_quaternion.shape}')  # NOQA
        if gt_translation is not None:
            logger.debug(f'gt_translation: {type(gt_translation)}, {gt_translation.shape}')  # NOQA

        batch_size = valid.shape[0]
        assert batch_size == 1
        valid = valid[0]

        if not valid:
            # skip invalid data
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        quaternion, translation = self.predict(
            class_id=class_id,
            pitch=pitch,
            cad_origin=cad_origin,
            cad_rgbs=cad_rgbs,
            cad_pcds=cad_pcds,
            scan_origin=scan_origin,
            scan_rgbs=scan_rgbs,
            scan_pcds=scan_pcds,
            scan_masks=scan_masks,
        )
        return self.loss(
            quaternion=quaternion,
            translation=translation,
            gt_quaternion=gt_quaternion,
            gt_translation=gt_translation,
        )

    def loss(self, quaternion, translation, gt_quaternion, gt_translation):
        logger.debug('==> Computing Loss')
        loss_quaternion = F.mean_absolute_error(quaternion, gt_quaternion)
        loss_translation = F.mean_absolute_error(translation, gt_translation)
        logger.debug(f'gt_quaternion: {gt_quaternion}, quaternion: {quaternion}')  # NOQA
        logger.debug(f'gt_translation: {gt_translation}, translation: {translation}')  # NOQA
        loss = (
            (self._lambda_quaternion * loss_quaternion) +
            (self._lambda_translation * loss_translation)
        )
        logger.debug(f'loss_quaternion: {loss_quaternion}')
        logger.debug(f'loss_translation: {loss_translation}')
        logger.debug(f'loss: {loss}')

        chainer.report(
            values={
                'loss_quaternion': loss_quaternion,
                'loss_translation': loss_translation,
                'loss': loss,
            },
            observer=self,
        )

        return loss
