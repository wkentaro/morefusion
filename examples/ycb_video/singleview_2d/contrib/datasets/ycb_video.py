import chainer
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp

from .base import DatasetBase


class YCBVideoDataset(DatasetBase):

    _root_dir = objslampp.datasets.YCBVideoDataset._root_dir

    def __init__(self, split, class_ids=None):
        super().__init__()
        self._split = split
        self._class_ids = class_ids
        self._ids = self._get_ids()

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

        if chainer.is_debug():
            print(f'[{index:08d}]: class_ids: {class_ids.tolist()}')

        examples = []
        for instance_id, class_id in enumerate(class_ids):
            if (self._class_ids is not None and
                    class_id not in self._class_ids):
                continue

            rgb = frame['color'].copy()
            mask = frame['label'] == class_id
            depth = frame['depth']
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
            mask = mask[y1:y2, x1:x2]
            pcd = pcd[y1:y2, x1:x2]

            rgb = imgviz.centerize(rgb, (256, 256))
            mask = imgviz.centerize(
                mask.astype(np.uint8), (256, 256)
            ).astype(bool)
            pcd = imgviz.centerize(pcd, (256, 256))

            rgb[~mask] = 0
            translation_rough = np.nanmean(pcd[mask], axis=0)

            T_cad2cam = frame['meta']['poses'][:, :, instance_id]
            quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
            translation_true = tf.translation_from_matrix(T_cad2cam)

            examples.append(dict(
                class_id=class_id,
                rgb=rgb,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
                translation_rough=translation_rough,
            ))
        return examples


if __name__ == '__main__':
    dataset = YCBVideoDataset('train', class_ids=[2])
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
