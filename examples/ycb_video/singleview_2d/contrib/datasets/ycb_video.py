import chainer
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp

from .base import DatasetBase


class YCBVideoDataset(DatasetBase):

    _root_dir = objslampp.datasets.YCBVideoDataset._root_dir

    def __init__(
        self,
        split,
        class_ids=None,
        augmentation={},
    ):
        super().__init__()
        self._split = split
        self._class_ids = class_ids
        self._ids = self._get_ids()
        self._augmentation = augmentation

    def _get_ids(self):
        assert self.split in ['train', 'syn', 'val']

        if self.split == 'val':
            ids = objslampp.datasets.YCBVideoDataset(
                split='keyframe'
            ).get_ids()
        elif self.split == 'train':
            ids = objslampp.datasets.YCBVideoDataset(
                split='train'
            ).get_ids(sampling=8)
        else:
            assert self.split == 'syn'
            ids = []

        ids = [(True, x) for x in ids]

        if self.split in ['train', 'syn']:
            ids_syn = objslampp.datasets.YCBVideoSyntheticDataset().get_ids()
            ids_syn = [(False, x) for x in ids_syn]
            ids += ids_syn

        return tuple(ids)

    def get_frame(self, index):
        is_real, image_id = self._ids[index]
        if is_real:
            frame = objslampp.datasets.YCBVideoDataset.get_frame(image_id)
        else:
            frame = objslampp.datasets.YCBVideoSyntheticDataset.get_frame(
                image_id
            )
        class_ids = frame['meta']['cls_indexes'].astype(int)

        return dict(
            rgb=frame['color'],
            instance_label=frame['label'],
            intrinsic_matrix=frame['meta']['intrinsic_matrix'],
        )

    def get_examples(self, index):
        is_real, image_id = self._ids[index]

        if is_real:
            frame = objslampp.datasets.YCBVideoDataset.get_frame(image_id)
        else:
            frame = objslampp.datasets.YCBVideoSyntheticDataset.get_frame(
                image_id
            )

        class_ids = frame['meta']['cls_indexes'].astype(int)

        if chainer.is_debug():
            print(f'[{index:08d}]: class_ids: {class_ids.tolist()}')

        if not is_real:
            if 2 not in class_ids:
                return []
            import pybullet
            import trimesh
            pybullet.connect(pybullet.DIRECT)
            models = objslampp.datasets.YCBVideoModels()
            for instance_id, class_id in enumerate(class_ids):
                cad_file = models.get_cad_model(class_id)
                T_cad2cam = np.r_[
                    frame['meta']['poses'][:, :, instance_id], [[0, 0, 0, 1]]
                ]
                objslampp.extra.pybullet.add_model(
                    cad_file,
                    position=tf.translation_from_matrix(T_cad2cam),
                    orientation=tf.quaternion_from_matrix(T_cad2cam)[[1, 2, 3, 0]],
                )
            K = frame['meta']['intrinsic_matrix']
            camera = trimesh.scene.Camera(
                focal=(K[0, 0], K[1, 1]),
                resolution=(frame['color'].shape[1], frame['color'].shape[0]),
            )
            rgb, depth, segm = objslampp.extra.pybullet.render_camera(
                np.eye(4),
                camera.fov[1],
                height=camera.resolution[1],
                width=camera.resolution[0],
            )
            pybullet.disconnect()
            label = np.zeros_like(frame['label'])
            for instance_id, class_id in enumerate(class_ids):
                label[segm == instance_id] = class_id
            frame['color'] = rgb
            frame['depth'] = depth
            frame['label'] = label

        examples = []
        for instance_id, class_id in enumerate(class_ids):
            if (self._class_ids is not None and
                    class_id not in self._class_ids):
                continue

            # get frame
            rgb = frame['color'].copy()
            depth = frame['depth']
            mask = frame['label'] == class_id
            if mask.sum() == 0:
                continue

            # augment
            if self._augmentation:
                rgb, depth, mask = self._augment(rgb, depth, mask)

            # masking
            rgb[~mask] = 0
            depth[~mask] = np.nan

            # get point cloud
            K = frame['meta']['intrinsic_matrix']
            pcd = objslampp.geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            )

            # crop
            bbox = objslampp.geometry.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
                continue
            rgb = rgb[y1:y2, x1:x2]
            pcd = pcd[y1:y2, x1:x2]

            # finalize
            rgb = imgviz.centerize(rgb, (256, 256))

            T_cad2cam = frame['meta']['poses'][:, :, instance_id]
            quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
            translation_true = tf.translation_from_matrix(T_cad2cam)
            translation_rough = np.nanmean(pcd, axis=(0, 1))

            examples.append(dict(
                class_id=class_id,
                rgb=rgb,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
                translation_rough=translation_rough,
            ))
        return examples


if __name__ == '__main__':
    dataset = YCBVideoDataset(
        'syn',
        class_ids=[2],
        augmentation={'rgb', 'depth', 'segm', 'occl'},
    )
    print(f'dataset_size: {len(dataset)}')

    def images():
        for i in range(0, len(dataset)):
            with chainer.using_config('debug', True):
                example = dataset[i]
            print(f'index: {i:08d}')
            print(f"class_id: {example['class_id']}")
            print(f"quaternion_true: {example['quaternion_true']}")
            print(f"translation_true: {example['translation_true']}")
            print(f"translation_rough: {example['translation_rough']}")
            if example['class_id'] > 0:
                yield imgviz.tile(
                    [dataset.get_frame(i)['rgb'], example['rgb']], (1, 2)
                )

    imgviz.io.pyglet_imshow(images())
    imgviz.io.pyglet_run()
