import chainer
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp

from .base import DatasetBase


class MySyntheticDataset(DatasetBase):

    def __init__(self, root_dir, class_ids=None):
        super().__init__()
        self._root_dir = root_dir
        self._class_ids = class_ids
        self._ids = self._get_ids()

    def _get_ids(self):
        ids = []
        for video_dir in sorted(self.root_dir.dirs()):
            for npz_file in sorted(video_dir.files()):
                frame_id = f'{npz_file.parent.name}/{npz_file.stem}'
                ids.append(frame_id)
        return ids

    def get_frame(self, index):
        frame_id = self.ids[index]
        npz_file = self.root_dir / f'{frame_id}.npz'
        frame = np.load(npz_file)

        instance_ids = frame['instance_ids']
        class_ids = frame['class_ids']
        Ts_cad2cam = frame['Ts_cad2cam']

        keep = class_ids > 0
        instance_ids = instance_ids[keep]
        class_ids = class_ids[keep]
        Ts_cad2cam = Ts_cad2cam[keep]

        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=frame['rgb'],
            depth=frame['depth'],
            instance_label=frame['instance_label'],
            intrinsic_matrix=frame['intrinsic_matrix'],
            T_cam2world=frame['T_cam2world'],
            Ts_cad2cam=Ts_cad2cam,
        )

    def get_examples(self, index):
        frame = self.get_frame(index)

        instance_ids = frame['instance_ids']
        class_ids = frame['class_ids']
        rgb = frame['rgb']
        depth = frame['depth']
        instance_label = frame['instance_label']
        K = frame['intrinsic_matrix']
        Ts_cad2cam = frame['Ts_cad2cam']

        if chainer.is_debug():
            print(f'[{index:08d}]: class_ids: {class_ids.tolist()}')
            print(f'[{index:08d}]: instance_ids: {instance_ids.tolist()}')

        examples = []
        for instance_id, class_id, T_cad2cam in zip(
            instance_ids, class_ids, Ts_cad2cam
        ):
            if (self._class_ids is not None and
                    class_id not in self._class_ids):
                continue

            mask = instance_label == instance_id
            bbox = objslampp.geometry.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
                continue

            rgb = frame['rgb'].copy()
            rgb[~mask] = 0
            rgb = rgb[y1:y2, x1:x2]
            rgb = imgviz.centerize(rgb, (256, 256))

            pcd = objslampp.geometry.pointcloud_from_depth(
                depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            )
            translation_rough = np.nanmean(pcd[mask], axis=0)

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
    root_dir = chainer.dataset.get_dataset_directory(
        'wkentaro/objslampp/ycb_video/synthetic_data/20190402_174648.841996'
    )
    dataset = MySyntheticDataset(root_dir, class_ids=[2])
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
