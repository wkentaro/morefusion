import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.vgg import VGG16
import numpy as np

# from .pspnet import PSPNetExtractor

import objslampp


class PoseNet(chainer.Chain):

    def __init__(
        self,
        *,
        n_fg_class,
        freeze_until,
        lambda_confidence,
        n_point=1000,
    ):
        super().__init__()
        with self.init_scope():
            # extractor
            self.vgg_extractor = VGG16(pretrained_model='imagenet')
            self.vgg_extractor.pick = [
                'conv4_3', 'conv3_3', 'conv2_2', 'conv1_2'
            ]
            self.vgg_extractor.remove_unused()
            self.conv5_extractor = L.Convolution2D(
                in_channels=512,
                out_channels=32,
                ksize=1,
                initialW=chainer.initializers.Normal(0.01),
            )
            # self.pspnet_extractor = PSPNetExtractor()
            self.posenet_extractor = PoseNetExtractor()
            # conv1
            self.conv1_rot = L.Convolution1D(1408, 640, 1)
            self.conv1_trans = L.Convolution1D(1408, 640, 1)
            self.conv1_conf = L.Convolution1D(1408, 640, 1)
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

        self._n_fg_class = n_fg_class
        self._freeze_until = freeze_until
        self._n_point = n_point
        self._lambda_confidence = lambda_confidence

    def predict(self, *, class_id, rgb, pcd):
        xp = self.xp

        B, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3
        assert rgb.dtype == np.uint8
        assert pcd.shape == (B, H, W, 3)
        assert pcd.dtype == np.float64

        mask = ~xp.isnan(pcd).any(axis=3)  # BHW

        # prepare
        rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)  # BHWC -> BCHW
        pcd = pcd.astype(np.float32).transpose(0, 3, 1, 2)  # BHWC -> BCHW

        mean = xp.asarray(self.vgg_extractor.mean)
        h_conv4_3, h_conv3_3, h_conv2_2, h_conv1_2 = \
            self.vgg_extractor(rgb - mean[None])
        if self._freeze_until == 'conv4_3':
            h_conv4_3.unchain_backward()
        elif self._freeze_until == 'conv3_3':
            h_conv3_3.unchain_backward()
        elif self._freeze_until == 'conv2_2':
            h_conv2_2.unchain_backward()
        elif self._freeze_until == 'conv1_2':
            h_conv1_2.unchain_backward()
        else:
            assert self._freeze_until == 'none'
        del h_conv3_3, h_conv2_2, h_conv1_2
        h_rgb = h_conv4_3  # 1/8, 256x256 -> 32x32
        # h_rgb = self.pspnet_extractor(h_rgb)  # 1/1
        h_rgb = F.relu(self.conv5_extractor(h_rgb))
        h_rgb = F.resize_images(h_rgb, (H, W))  # 1/1

        # extract indices
        h_rgb_masked = []
        pcd_masked = []
        centroids = []
        for i in range(B):
            n_point = int(mask[i].sum())
            if n_point >= self._n_point:
                keep = xp.random.permutation(n_point)[:self._n_point]
            else:
                keep = xp.r_[
                    xp.arange(n_point),
                    xp.random.randint(0, n_point, self._n_point - n_point),
                ]
            assert keep.shape == (self._n_point,)
            iy, ix = xp.where(mask[i])
            centroid = pcd[i, :, iy, ix].mean(axis=0)
            iy, ix = iy[keep], ix[keep]
            h_rgb_i = h_rgb[i, :, iy, ix]      # CHW -> MC, M = self._n_point
            pcd_i = pcd[i, :, iy, ix]          # CHW -> MC
            h_rgb_i = h_rgb_i.transpose(1, 0)  # MC -> CM
            pcd_i = pcd_i.transpose(1, 0)      # MC -> CM
            h_rgb_masked.append(h_rgb_i[None])
            pcd_masked.append(pcd_i[None])
            centroids.append(centroid[None])
        h_rgb_masked = F.concat(h_rgb_masked, axis=0)
        pcd_masked = xp.concatenate(pcd_masked, axis=0)
        centroids = xp.concatenate(centroids, axis=0)

        pcd_masked = pcd_masked - centroids[:, :, None]
        h = self.posenet_extractor(h_rgb_masked, pcd_masked)

        # conv1
        h_rot = F.relu(self.conv1_rot(h))
        h_trans = F.relu(self.conv1_trans(h))
        h_conf = F.relu(self.conv1_conf(h))

        # conv2
        h_rot = F.relu(self.conv2_rot(h_rot))
        h_trans = F.relu(self.conv2_trans(h_trans))
        h_conf = F.relu(self.conv2_conf(h_conf))
        # conv3
        h_rot = F.relu(self.conv3_rot(h_rot))
        h_trans = F.relu(self.conv3_trans(h_trans))
        h_conf = F.relu(self.conv3_conf(h_conf))
        # conv4
        cls_rot = self.conv4_rot(h_rot)
        cls_trans = self.conv4_trans(h_trans)
        cls_conf = F.sigmoid(self.conv4_conf(h_conf))

        cls_rot = cls_rot.reshape((B, self._n_fg_class, 4, self._n_point))
        cls_trans = cls_trans.reshape((B, self._n_fg_class, 3, self._n_point))
        cls_conf = cls_conf.reshape((B, self._n_fg_class, self._n_point))

        pcd_masked = pcd_masked + centroids[:, :, None]
        cls_trans = pcd_masked[:, None, :, :] + cls_trans

        rot = cls_rot[xp.arange(B), class_id, :, :]
        trans = cls_trans[xp.arange(B), class_id, :, :]
        conf = cls_conf[xp.arange(B), class_id]

        rot = F.normalize(rot, axis=1)
        rot = rot.transpose(0, 2, 1)    # B4M -> BM4
        trans = trans.transpose(0, 2, 1)  # B3M -> BM3

        return rot, trans, conf

    def __call__(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        keep = class_id != -1
        if keep.sum() == 0:
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        class_id = class_id[keep]
        rgb = rgb[keep]
        pcd = pcd[keep]
        quaternion_true = quaternion_true[keep]
        translation_true = translation_true[keep]

        quaternion_pred, translation_pred, confidence_pred = self.predict(
            class_id=class_id, rgb=rgb, pcd=pcd
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


# https://github.com/knorth55/chainer-dense-fusion/blob/master/chainer_dense_fusion/links/model/posenet.py  # NOQA
class PoseNetExtractor(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # conv1
            self.conv1_rgb = L.Convolution1D(32, 64, 1)
            self.conv1_pcd = L.Convolution1D(3, 64, 1)
            # conv2
            self.conv2_rgb = L.Convolution1D(64, 128, 1)
            self.conv2_pcd = L.Convolution1D(64, 128, 1)
            # conv3, conv4
            self.conv3 = L.Convolution1D(256, 512, 1)
            self.conv4 = L.Convolution1D(512, 1024, 1)

    def __call__(self, h_rgb, pcd):
        B, _, n_point = h_rgb.shape
        # conv1
        h_rgb = F.relu(self.conv1_rgb(h_rgb))
        h_pcd = F.relu(self.conv1_pcd(pcd))
        feat1 = F.concat((h_rgb, h_pcd), axis=1)
        # conv2
        h_rgb = F.relu(self.conv2_rgb(h_rgb))
        h_pcd = F.relu(self.conv2_pcd(h_pcd))
        feat2 = F.concat((h_rgb, h_pcd), axis=1)
        # conv3, conv4
        h = F.relu(self.conv3(feat2))
        h = F.relu(self.conv4(h))
        h = F.average_pooling_1d(h, n_point)
        h = h.reshape((B, 1024, 1))
        feat3 = F.repeat(h, n_point, axis=2)
        feat = F.concat((feat1, feat2, feat3), axis=1)
        return feat


if __name__ == '__main__':
    from .dataset import Dataset

    gpu = 0

    dataset = Dataset(split='train', class_ids=[2])
    inputs = chainer.dataset.concat_examples(dataset[0:2], device=gpu)

    model = PoseNet(
        n_fg_class=21,
        freeze_until='conv4_3',
        lambda_confidence=0.015,
    )
    if gpu >= -1:
        model.to_gpu()

    loss = model.__call__(**inputs)
