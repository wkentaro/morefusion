import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.resnet import ResNet50
from chainercv.links.model.vgg import VGG16
import imgviz
import numpy as np
import trimesh.transformations as tf

from ..logger import logger
from .. import extra
from .. import geometry
from .. import functions
from .. import metrics
from .. import training


class MultiViewAlignmentModel(chainer.Chain):

    def __init__(
        self,
        *,
        extractor,
        lambda_quaternion=1.0,
        lambda_translation=1.0,
        writer=None,
        write_interval=100,
        loss_function='l1',
    ):
        super(MultiViewAlignmentModel, self).__init__()

        self._lambda_quaternion = lambda_quaternion
        self._lambda_translation = lambda_translation

        if writer is None:
            writer = training.SummaryWriterWithUpdater(writer=None)  # dummy
            write_interval = 0
        self._writer = writer
        self._write_interval = write_interval

        assert loss_function in ['l1', 'add', 'add_rotation']
        self._loss_function = loss_function

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

    def trigger_write(self):
        return (
            chainer.config.train and
            (self._write_interval > 0) and
            (self._writer.global_step % self._write_interval == 1)
        )

    def encode(
        self,
        origin,
        pitch,
        rgbs,
        pcds,
        masks=None,
        return_fused=False,
    ):
        assert origin.shape == (3,)
        assert pitch.shape == ()

        N, H, W, C = rgbs.shape
        assert C == 3
        assert pcds.shape == (N, H, W, 3)

        if masks is None:
            masks = ~self.xp.isnan(pcds).any(axis=3)
            assert masks.shape == (N, H, W)

        if self.trigger_write():
            self._writer.add_image(
                'rgbs',
                imgviz.tile(cuda.to_cpu(rgbs), border=(255, 255, 255)),
                dataformats='HWC',
            )

            depth2rgb = imgviz.Depth2RGB()
            self._writer.add_image(
                'pcds',
                imgviz.tile(
                    [depth2rgb(x) for x in cuda.to_cpu(pcds[:, :, :, 2])],
                    border=(255, 255, 255),
                ),
                dataformats='HWC',
            )
            del depth2rgb

            self._writer.add_image(
                'masks',
                imgviz.tile(
                    [x.astype(np.uint8) * 255 for x in cuda.to_cpu(masks)],
                    border=(255, 255, 255),
                ),
                dataformats='HWC',
            )

        # MV
        mean = self.xp.asarray(self.extractor.mean)
        rgbs = rgbs.transpose(0, 3, 1, 2).astype(np.float32)
        rgbs = rgbs - mean[None]
        if isinstance(self.extractor, ResNet50):
            with chainer.using_config('train', False):
                h, = self.extractor(rgbs)
        else:
            assert isinstance(self.extractor, VGG16)
            h, = self.extractor(rgbs)
        if chainer.is_debug():
            logger.info(f'h_extractor: {h.shape}')

        h = F.relu(self.conv5(h))
        if chainer.is_debug():
            logger.info(f'h_conv5: {h.shape}')

        isnans = self.xp.isnan(pcds).any(axis=3)
        pcds[isnans] = 0

        masks = (~isnans) & masks
        bboxes = geometry.masks_to_bboxes(chainer.cuda.to_cpu(masks))
        h = F.resize_images(h, rgbs.shape[2:4])

        if self.trigger_write():
            nchannel2rgb = imgviz.Nchannel2RGB()
            self._writer.add_image(
                'feature_2d',
                imgviz.tile([
                    nchannel2rgb(x)
                    for x in cuda.to_cpu(h.array).transpose(0, 2, 3, 1)
                ], border=(255, 255, 255)),
                dataformats='HWC',
            )
            self._writer.add_image(
                'feature_2d_masked',
                imgviz.tile([
                    nchannel2rgb(x) * m[:, :, None]
                    for x, m in zip(
                        cuda.to_cpu(h.array).transpose(0, 2, 3, 1),
                        cuda.to_cpu(masks),
                    )
                ], border=(255, 255, 255)),
                dataformats='HWC',
            )
            del nchannel2rgb

            self._writer.add_histogram(
                'feature_2d',
                cuda.to_cpu(h.array),
            )

        if self.trigger_write():
            images = []
            nchannel2rgb = imgviz.Nchannel2RGB()
            depth2rgb = imgviz.Depth2RGB()

        h_vox = []
        for i in range(N):
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

            output_shape = (128, 128)
            h_i = F.resize_images(h_i[None, :, :, :], output_shape)[0]
            pcd = extra.cupy.resize_image(pcd, output_shape, order='HWC')
            mask = extra.cupy.resize_image(mask, output_shape, order='HW')

            if self.trigger_write():
                image = imgviz.tile([
                    nchannel2rgb(cuda.to_cpu(h_i.array.transpose(1, 2, 0))),
                    depth2rgb(cuda.to_cpu(pcd[:, :, 2])),
                    imgviz.gray2rgb(cuda.to_cpu(mask).astype(np.uint8) * 255)
                ], border=(255, 255, 255))
                images.append(image)
                del image

            h_i = h_i.transpose(1, 2, 0)  # CHW -> HWC
            h_i = functions.voxelization_3d(
                values=h_i[mask, :],
                points=pcd[mask, :],
                origin=origin,
                pitch=pitch,
                dimensions=(self.voxel_dim,) * 3,
                channels=self.voxel_channels,
            )  # CXYZ
            if chainer.is_debug():
                logger.info(f'h_i, i={i}: {h_i.shape}')
            h_vox.append(h_i[None])
        h = F.concat(h_vox, axis=0)
        if chainer.is_debug():
            logger.info(f'h_vox: {h.shape}')
        h = F.max(h, axis=0)[None]
        if chainer.is_debug():
            logger.info(f'h_vox_fused: {h.shape}')  # NOQA

        if self.trigger_write():
            self._writer.add_image(
                'feature_2d_roi',
                imgviz.tile(images, border=(255, 255, 255)),
                dataformats='HWC',
            )
            del images, nchannel2rgb, depth2rgb

            self._writer.add_histogram(
                'feature_voxel',
                cuda.to_cpu(h.array),
            )

        if return_fused:
            return h

        # 3DCNN
        h = F.relu(self.conv6(h))
        if chainer.is_debug():
            logger.info(f'h_conv6: {h.shape}')
        h = F.relu(self.conv7(h))
        if chainer.is_debug():
            logger.info(f'h_conv7: {h.shape}')
        return h

    def predict_from_code(self, h_cad, h_scan):
        h = F.concat([h_cad, h_scan], axis=1)
        if chainer.is_debug():
            logger.info(f'h_concat: {h.shape}')

        h = F.relu(self.fc8(h))
        if chainer.is_debug():
            logger.info(f'h_fc8: {h.shape}')

        quaternion = F.normalize(self.fc_quaternion(h))  # [-1, 1]
        if chainer.is_debug():
            logger.info(f'quaternion: {quaternion}')
        translation = F.cos(self.fc_translation(h))      # [-1, 1]
        if chainer.is_debug():
            logger.info(f'translation: {translation}')
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
        cad_origin = cad_origin[0]
        cad_rgbs = cad_rgbs[0]
        cad_pcds = cad_pcds[0]
        scan_origin = scan_origin[0]
        scan_rgbs = scan_rgbs[0]
        scan_pcds = scan_pcds[0]
        scan_masks = scan_masks[0]

        assert class_id > 0  # 0 indicates background class

        if chainer.is_debug():
            print('==> Multi-View Encoding CAD')
        with self._writer.scope('cad'):
            h_cad = self.encode(
                origin=cad_origin,
                pitch=pitch,
                rgbs=cad_rgbs,
                pcds=cad_pcds,
            )

        if chainer.is_debug():
            print('==> Multi-View Encoding Scan')
        with self._writer.scope('scan'):
            h_scan = self.encode(
                origin=scan_origin,
                pitch=pitch,
                rgbs=scan_rgbs,
                pcds=scan_pcds,
                masks=scan_masks
            )

        if chainer.is_debug():
            print('==> Predicting Pose')
        quaternion, translation = self.predict_from_code(h_cad, h_scan)

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
        cad_points=None,  # for evaluation
    ):
        xp = self.xp

        if chainer.is_debug():
            print('==> Arguments for MultiViewAlignmentModel.__call__')
            logger.info(f'valid: {type(valid)}, {valid}')
            logger.info(f'video_id: {type(video_id)}, {video_id}')
            logger.info(f'class_id: {type(class_id)}, {class_id}')
            logger.info(f'pitch: {type(pitch)}, {pitch}')
            logger.info(f'cad_origin: {type(cad_origin)}, {cad_origin}')
            logger.info(f'cad_rgbs: {type(cad_rgbs)}, {cad_rgbs.shape}')
            logger.info(f'cad_pcds: {type(cad_pcds)}, {cad_pcds.shape}')
            logger.info(
                f'scan_origin: {type(scan_origin)}, {scan_origin.shape}'
            )
            logger.info(f'scan_rgbs: {type(scan_rgbs)}, {scan_rgbs.shape}')
            logger.info(f'scan_pcds: {type(scan_pcds)}, {scan_pcds.shape}')
            logger.info(f'scan_masks: {type(scan_masks)}, {scan_masks.shape}')
            if gt_pose is not None:
                logger.info(f'gt_pose: {type(gt_pose)}, {gt_pose.shape}')
            if gt_quaternion is not None:
                logger.info(
                    f'gt_quaternion: {type(gt_quaternion)}, {gt_quaternion}'
                )
            if gt_translation is not None:
                logger.info(
                    f'gt_translation: {type(gt_translation)}, {gt_translation}'
                )

        batch_size = valid.shape[0]
        assert batch_size == 1
        valid = valid[0]

        if not valid:
            # skip invalid data
            return chainer.Variable(xp.zeros((), dtype=np.float32))

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

        if cad_points is not None:
            self.evaluate(
                class_id=class_id,
                pitch=pitch,
                scan_origin=scan_origin,
                cad_origin=cad_origin,
                cad_points=cad_points,
                quaternion=quaternion,
                translation=translation,
                gt_quaternion=gt_quaternion,
                gt_translation=gt_translation,
            )

        return self.loss(
            quaternion=quaternion,
            translation=translation,
            gt_quaternion=gt_quaternion,
            gt_translation=gt_translation,
            video_id=video_id,
            cad_points=cad_points,
            pitch=pitch,
            cad_origin=cad_origin,
            scan_origin=scan_origin,
        )

    def evaluate(
        self,
        *,
        class_id,
        pitch,
        scan_origin,
        cad_origin,
        cad_points,
        quaternion,
        translation,
        gt_quaternion,
        gt_translation,
    ):
        batch_size = quaternion.shape[0]
        assert batch_size == 1

        class_id = int(class_id[0])
        pitch = float(pitch[0])
        scan_origin = cuda.to_cpu(scan_origin[0])
        cad_origin = cuda.to_cpu(cad_origin[0])
        cad_points = cuda.to_cpu(cad_points[0])
        quaternion = cuda.to_cpu(quaternion.array[0])
        translation = cuda.to_cpu(translation.array[0])
        gt_quaternion = cuda.to_cpu(gt_quaternion[0])
        gt_translation = cuda.to_cpu(gt_translation[0])

        transform_pred = tf.quaternion_matrix(quaternion)
        transform_true = tf.quaternion_matrix(gt_quaternion)
        add_rotation = metrics.average_distance(
            [cad_points], [transform_true], [transform_pred]
        )[0]

        transform_pred[:3, 3] = (
            (scan_origin - cad_origin) +
            translation * self.voxel_dim * pitch
        )
        transform_true[:3, 3] = (
            (scan_origin - cad_origin) +
            gt_translation * self.voxel_dim * pitch
        )
        add = metrics.average_distance(
            [cad_points], [transform_true], [transform_pred]
        )[0]

        values = {
            f'add_rotation': add_rotation,
            f'add': add,
        }
        if not chainer.config.train:
            for key in list(values.keys()):
                values[f'{key}/{class_id:04d}'] = values.pop(key)
        chainer.report(values, observer=self)

    def loss(
        self,
        quaternion,
        translation,
        gt_quaternion,
        gt_translation,
        *,
        video_id=None,
        cad_points=None,
        pitch=None,
        cad_origin=None,
        scan_origin=None,
    ):
        batch_size = quaternion.shape[0]
        assert batch_size == 1
        if video_id is not None:
            video_id = int(video_id[0])

        if chainer.is_debug():
            print('==> Computing Loss')
        if self._loss_function == 'l1':
            loss_quaternion = F.mean_absolute_error(
                quaternion, gt_quaternion
            )
            loss_translation = F.mean_absolute_error(
                translation, gt_translation
            )
            loss = (
                (self._lambda_quaternion * loss_quaternion) +
                (self._lambda_translation * loss_translation)
            )
        else:
            if self._loss_function == 'add':
                translation_true = (
                    (scan_origin - cad_origin) +
                    (gt_translation * self.voxel_dim * pitch)
                )[0]
                translation_pred = (
                    (scan_origin - cad_origin) +
                    (translation * self.voxel_dim * pitch)
                )[0]
            else:
                assert self._loss_function == 'add_rotation'
                translation_true = self.xp.zeros((3,), dtype=np.float32)
                translation_pred = self.xp.zeros((3,), dtype=np.float32)

            transform_true = functions.quaternion_matrix(gt_quaternion[0])
            transform_pred = functions.quaternion_matrix(quaternion[0])

            assert cad_points is not None
            loss = functions.average_distance(
                points=cad_points[0],
                transform1=transform_true,
                transform2=transform_pred,
                translation1=translation_true,
                translation2=translation_pred
            )
            loss_quaternion = 0
            loss_translation = 0
        if chainer.is_debug():
            logger.info(
                f'gt_quaternion: {gt_quaternion}, quaternion: {quaternion}'
            )
            logger.info(
                f'gt_translation: {gt_translation}, translation: {translation}'
            )
            logger.info(f'loss_quaternion: {loss_quaternion}')
            logger.info(f'loss_translation: {loss_translation}')
            logger.info(f'loss: {loss}')

        values = {
            'loss_quaternion': loss_quaternion,
            'loss_translation': loss_translation,
            'loss': loss,
        }
        if not chainer.config.train and video_id is not None:
            for key in list(values.keys()):
                values[f'{key}/{video_id:04d}'] = values.pop(key)
        chainer.report(values=values, observer=self)

        return loss
