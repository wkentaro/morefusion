#!/usr/bin/env python

import pathlib

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.vgg import VGG16
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp


class Dataset(objslampp.datasets.YCBVideoDataset):

    def __init__(self, split, class_ids=None):
        self._class_ids = class_ids
        super(Dataset, self).__init__(split=split, sampling=15)

    def get_ids(
        self,
        split: str,
        sampling: int = 1,
    ):
        assert split in ('train', 'val')

        video2class_ids: dict = {}
        imageset_file: pathlib.Path = self.root_dir / f'image_sets/{split}.txt'
        with open(imageset_file) as f:
            ids: list = []
            for line in f:
                image_id = line.strip()
                video_id, frame_id = image_id.split('/')
                if int(frame_id) % sampling == 0:
                    if video_id in video2class_ids:
                        class_ids = video2class_ids[video_id]
                    else:
                        frame = self.get_frame(image_id)
                        class_ids = frame['meta']['cls_indexes']
                        video2class_ids[video_id] = class_ids
                    ids += [
                        (image_id, class_id) for class_id in class_ids
                        if self._class_ids is None or
                        class_id in self._class_ids
                    ]
            return tuple(ids)

    def get_example(self, index):
        image_id, class_id = self._ids[index]
        frame = self.get_frame(image_id)

        class_ids = frame['meta']['cls_indexes']
        assert class_id in class_ids
        instance_id = np.where(class_ids == class_id)[0][0]

        mask = frame['label'] == class_id
        bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
        y1, x1, y2, x2 = bbox.round().astype(int)

        rgb = frame['color'].copy()
        rgb[~mask] = 0
        rgb = rgb[y1:y2, x1:x2]
        rgb = imgviz.centerize(rgb, (256, 256))

        depth = frame['depth']
        K = frame['meta']['intrinsic_matrix']
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )
        translation_rough = np.nanmean(pcd[mask], axis=0)

        T_cad2cam = frame['meta']['poses'][:, :, instance_id]
        quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
        translation_true = tf.translation_from_matrix(T_cad2cam)

        model_dataset = objslampp.datasets.YCBVideoModelsDataset()
        cad_pcd_file = model_dataset.get_model(class_id=class_id)['points_xyz']
        cad_pcd = np.loadtxt(cad_pcd_file)

        return (
            cad_pcd,
            rgb,
            quaternion_true,
            translation_true,
            translation_rough,
        )


class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()

        initialW = chainer.initializers.Normal(0.01)
        with self.init_scope():
            self.extractor = VGG16(pretrained_model='imagenet')
            self.extractor.pick = ['pool4']
            self.extractor.remove_unused()
            self.fc_quaternion = L.Linear(512, 4, initialW=initialW)

    def evaluate(
        self,
        cad_pcd,
        quaternion_true,
        quaternion_pred,
        translation_true,
        translation_rough,
    ):
        assert quaternion_pred.shape[0] == 1
        cad_pcd = cuda.to_cpu(cad_pcd[0])
        quaternion_true = cuda.to_cpu(quaternion_true[0])
        quaternion_pred = cuda.to_cpu(quaternion_pred.array[0])
        translation_true = cuda.to_cpu(translation_true[0])
        translation_rough = cuda.to_cpu(translation_rough[0])

        T_cam2cad_true = tf.quaternion_matrix(quaternion_true)
        T_cam2cad_pred = tf.quaternion_matrix(quaternion_pred)
        add_rotation = objslampp.metrics.average_distance(
            [cad_pcd], [T_cam2cad_true], [T_cam2cad_pred]
        )[0]

        T_cam2cad_true = objslampp.geometry.compose_transform(
            R=T_cam2cad_true[:3, :3], t=translation_true
        )
        T_cam2cad_pred = objslampp.geometry.compose_transform(
            R=T_cam2cad_pred[:3, :3], t=translation_rough
        )
        add = objslampp.metrics.average_distance(
            [cad_pcd], [T_cam2cad_true], [T_cam2cad_pred]
        )[0]

        if chainer.config.train:
            values = {
                'add': add,
                'add_rotation': add_rotation,
            }
        else:
            values = {
                'add/0002': add,
                'add_rotation/0002': add_rotation,
            }
        chainer.report(values, observer=self)

    def __call__(
        self,
        cad_pcd,
        rgb,
        quaternion_true,
        translation_true,
        translation_rough,
    ):
        xp = self.xp

        assert rgb.ndim == 4
        N, H, W, C = rgb.shape
        assert H == W == 256
        assert C == 3

        rgb = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW
        quaternion_true = quaternion_true.astype(np.float32)
        cad_pcd = cad_pcd.astype(np.float32)

        mean = xp.asarray(self.extractor.mean)
        h, = self.extractor(rgb - mean[None])  # NCHW
        h = F.average(h, axis=(2, 3))

        quaternion = F.normalize(self.fc_quaternion(h))

        self.evaluate(
            cad_pcd,
            quaternion_true,
            quaternion,
            translation_true,
            translation_rough,
        )

        T_cam2cad_pred = objslampp.functions.quaternion_matrix(
            quaternion
        )
        T_cam2cad_true = objslampp.functions.quaternion_matrix(
            quaternion_true
        )

        assert cad_pcd.shape[0] == 1
        loss = objslampp.functions.average_distance(
            cad_pcd[0], T_cam2cad_true[0], T_cam2cad_pred[0]
        )

        if chainer.config.train:
            values = {'loss': loss}
        else:
            values = {'loss/0002': loss}
        chainer.report(values, observer=self)

        return loss


if __name__ == '__main__':
    import trimesh

    dataset = Dataset('train', class_ids=[2])
    cad_pcd, rgb, quaternion_true, translation_true, translation_rough = \
        dataset[1]

    if 1:
        # trimesh.PointCloud(cad_pcd).show()
        imgviz.io.pyglet_imshow(rgb, caption='input image')

        model_dataset = objslampp.datasets.YCBVideoModelsDataset()
        cad_file = model_dataset.get_model(class_id=2)['textured_simple']
        cad = trimesh.load(str(cad_file))

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.axis())  # camera axis

        cad_true = cad.copy()
        T_cam2cad_true = tf.quaternion_matrix(quaternion_true)
        T_cam2cad_true = objslampp.geometry.compose_transform(
            T_cam2cad_true[:3, :3], translation_true
        )
        cad_true.apply_transform(T_cam2cad_true)
        scene.add_geometry(cad_true)
        del cad_true, T_cam2cad_true

        cad_rough = cad.copy()
        T_cam2cad_rough = tf.quaternion_matrix(quaternion_true)
        T_cam2cad_rough = objslampp.geometry.compose_transform(
            T_cam2cad_rough[:3, :3], translation_rough
        )
        cad_rough.apply_transform(T_cam2cad_rough)
        scene.add_geometry(cad_rough)
        del cad_rough, T_cam2cad_rough

        scene.show(caption='transform cad->camera', resolution=(512, 512))

        imgviz.io.pyglet_run()

    model = Model()
    loss = model(
        cad_pcd[None],
        rgb[None],
        quaternion_true[None],
        translation_true[None],
        translation_rough[None],
    )
