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

        augmentation_all = {'rgb', 'depth', 'segm', 'occl'}
        assert augmentation_all.issuperset(set(augmentation))
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
        elif self.split == 'syn':
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

        class_ids = frame['meta']['cls_indexes']

        examples = []
        for class_id in class_ids:
            rgb = frame['color'].copy()
            depth = frame['depth'].copy()
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
            pcd = imgviz.centerize(pcd, (256, 256), cval=np.nan)
            if np.isnan(pcd).any(axis=2).all():
                continue

            instance_id = np.where(class_ids == class_id)[0][0]
            T_cad2cam = frame['meta']['poses'][:, :, instance_id]
            quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
            translation_true = tf.translation_from_matrix(T_cad2cam)

            examples.append(dict(
                class_id=class_id,
                pitch=self._get_pitch(class_id=class_id),
                rgb=rgb,
                pcd=pcd,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
            ))
        return examples


if __name__ == '__main__':
    dataset = YCBVideoDataset(
        'train',
        class_ids=[2],
        augmentation={'rgb', 'depth', 'segm', 'occl'},
    )
    print(f'dataset_size: {len(dataset)}')

    def images():
        for i in range(0, len(dataset)):
            example = dataset[i]
            print(f"class_id: {example['class_id']}")
            print(f"pitch: {example['pitch']}")
            print(f"quaternion_true: {example['quaternion_true']}")
            print(f"translation_true: {example['translation_true']}")
            if example['class_id'] > 0:
                viz = imgviz.tile([
                    example['rgb'],
                    imgviz.depth2rgb(example['pcd'][:, :, 0]),
                    imgviz.depth2rgb(example['pcd'][:, :, 1]),
                    imgviz.depth2rgb(example['pcd'][:, :, 2]),
                ], (1, 4), border=(255, 255, 255))
                yield viz

    imgviz.io.pyglet_imshow(images())
    imgviz.io.pyglet_run()
