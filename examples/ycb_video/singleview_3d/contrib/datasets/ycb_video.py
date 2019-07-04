import pickle

import chainer
import numpy as np
import path

import objslampp

from .base import DatasetBase


class YCBVideoDataset(DatasetBase):

    _root_dir = objslampp.datasets.YCBVideoDataset._root_dir
    _cache_dir = chainer.dataset.get_dataset_directory(
        'wkentaro/objslampp/ycb_video/singleview_3d/ycb_video/cache_examples'
    )
    _cache_dir = path.Path(_cache_dir)

    def __init__(
        self,
        split,
        class_ids=None,
        augmentation=None,
        return_occupancy_grids=False,
        sampling=None,
    ):
        super().__init__(
            class_ids=class_ids,
            augmentation=augmentation,
            return_occupancy_grids=return_occupancy_grids,
        )
        self._split = split
        self._sampling = sampling
        self._ids = self._get_ids()

    def get_examples(self, index):
        is_real, image_id = self._ids[index]
        if is_real:
            cache_file = self._cache_dir / 'real' / f'{image_id}.pkl'
        else:
            cache_file = self._cache_dir / 'syn' / f'{image_id}.pkl'
        cache_file.parent.makedirs_p()

        examples = None

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    examples = pickle.load(f)
                if not self._return_occupancy_grids:
                    examples.pop('grid_target')
                    examples.pop('grid_nontarget')
                    examples.pop('grid_empty')
            except Exception:
                pass

        if examples is None:
            examples = super().get_examples(index)
            if self._return_occupancy_grids:
                with open(cache_file, 'wb') as f:
                    pickle.dump(examples, f)

        assert examples is not None
        return examples

    def _get_ids(self):
        assert self.split in ['train', 'syn', 'val']

        if self.split == 'val':
            sampling = 1 if self._sampling is None else self._sampling
            ids = objslampp.datasets.YCBVideoDataset(
                split='keyframe'
            ).get_ids(sampling=sampling)
        elif self.split == 'train':
            sampling = 8 if self._sampling is None else self._sampling
            ids = objslampp.datasets.YCBVideoDataset(
                split='train'
            ).get_ids(sampling=sampling)
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
        class_ids = frame['meta']['cls_indexes'].astype(np.int32)
        instance_ids = class_ids.copy()
        if 'rotation_translation_matrix' in frame['meta']:
            T_cam2world = frame['meta']['rotation_translation_matrix']
            T_cam2world = np.r_[T_cam2world, [[0, 0, 0, 1]]].astype(float)
        else:
            T_cam2world = np.eye(4, dtype=float)
        n_instance = len(instance_ids)
        Ts_cad2cam = np.zeros((n_instance, 4, 4), dtype=float)
        for i in range(n_instance):
            T_cad2cam = frame['meta']['poses'][:, :, i]
            T_cad2cam = np.r_[T_cad2cam, [[0, 0, 0, 1]]]
            Ts_cad2cam[i] = T_cad2cam
        return dict(
            instance_ids=instance_ids,
            class_ids=class_ids,
            rgb=frame['color'],
            depth=frame['depth'],
            instance_label=frame['label'],
            intrinsic_matrix=frame['meta']['intrinsic_matrix'],
            T_cam2world=T_cam2world,
            Ts_cad2cam=Ts_cad2cam,
            cad_files={},
        )


def main():
    dataset = YCBVideoDataset(
        'train',
        class_ids=None,
        augmentation={'rgb', 'depth'},
        return_occupancy_grids=True,
    )
    print(f'dataset_size: {len(dataset)}')

    # -------------------------------------------------------------------------

    import imgviz

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


if __name__ == '__main__':
    main()
