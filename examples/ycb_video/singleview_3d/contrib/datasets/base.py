import tempfile

import chainer
import filelock
import imgaug
import imgaug.augmenters as iaa
import imgviz
import numpy as np
import path
import trimesh.transformations as tf

import objslampp

from .multi_instance_octree_mapping import MultiInstanceOctreeMapping


class DatasetBase(objslampp.datasets.DatasetBase):

    _models = objslampp.datasets.YCBVideoModels()
    _voxel_dim = 32
    _mask_size_minimal = 1

    def __init__(
        self,
        root_dir=None,
        class_ids=None,
        augmentation=None,
        return_occupancy_grids=False,
    ):
        self._root_dir = root_dir
        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids
        self._augmentation = augmentation
        self._return_occupancy_grids = return_occupancy_grids

    def _get_cache_dir(self, index):
        raise NotImplementedError

    def _get_invalid_data(self):
        example = dict(
            class_id=-1,
            pitch=0.,
            origin=np.zeros((3,), dtype=np.float64),
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            pcd=np.zeros((256, 256, 3), dtype=np.float64),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
        )
        if self._return_occupancy_grids:
            dimensions = (self._voxel_dim,) * 3
            example['grid_target'] = np.zeros(dimensions, dtype=np.float64)
            example['grid_nontarget'] = np.zeros(dimensions, dtype=np.float64)
            example['grid_empty'] = np.zeros(dimensions, dtype=np.float64)
        return example

    def _get_pitch(self, class_id):
        return self._models.get_voxel_pitch(
            dimension=self._voxel_dim, class_id=class_id
        )

    def get_examples(self, index):
        if self._augmentation:
            return self._get_examples(index, filter_class_ids=True)

        cache_dir = self._get_cache_dir(index)

        examples = None

        if cache_dir.exists():
            try:
                examples = []
                for file in sorted(cache_dir.glob('*.npz')):
                    example = dict(np.load(file))
                    for k in example.keys():
                        if example[k].shape == ():
                            example[k] = example[k].item()
                    if not self._return_occupancy_grids:
                        example.pop('grid_target')
                        example.pop('grid_nontarget')
                        example.pop('grid_empty')
                    examples.append(example)
            except Exception:
                examples = None
                cache_dir.rmtree()

        if examples is None:
            if self._return_occupancy_grids:
                examples = self._get_examples(index)

                cache_dir.parent.makedirs_p()

                temp_dir = tempfile.mkdtemp(
                    prefix=cache_dir.name, dir=cache_dir.parent
                )
                temp_dir = path.Path(temp_dir)
                for i, example in enumerate(examples):
                    assert 'grid_target' in example
                    assert 'grid_nontarget' in example
                    assert 'grid_empty' in example
                    file = temp_dir / f'{i:04d}.npz'
                    np.savez_compressed(file, **example)

                lock_path = cache_dir + '_lock'
                lock_path.parent.makedirs_p()
                with filelock.FileLock(lock_path):
                    cache_dir.parent.makedirs_p()
                    temp_dir.move(cache_dir)
                temp_dir.rmtree_p()
            else:
                examples = self._get_examples(index, filter_class_ids=True)

        assert examples is not None
        return examples

    def _get_examples(self, index, filter_class_ids=False):
        frame = self.get_frame(index)

        instance_ids = frame['instance_ids']
        class_ids = frame['class_ids']
        rgb = frame['rgb']
        depth = frame['depth']
        instance_label = frame['instance_label']
        K = frame['intrinsic_matrix']
        Ts_cad2cam = frame['Ts_cad2cam']
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        if instance_ids.size == 0:
            return [self._get_invalid_data()]

        if chainer.is_debug():
            print(f'[{index:08d}]: class_ids: {class_ids.tolist()}')
            print(f'[{index:08d}]: instance_ids: {instance_ids.tolist()}')

        if self._return_occupancy_grids:
            H = pcd.shape[0]
            pcd_2s = imgviz.resize(pcd, height=H // 2, interpolation='nearest')
            instance_label_2s = imgviz.resize(
                instance_label, height=H // 2, interpolation='nearest'
            )
            pcd_4s = imgviz.resize(pcd, height=H // 4, interpolation='nearest')
            instance_label_4s = imgviz.resize(
                instance_label, height=H // 4, interpolation='nearest'
            )

            mapping = MultiInstanceOctreeMapping()

            nonnan = ~np.isnan(pcd_2s).any(axis=2)
            masks = np.array([
                (instance_label_2s == i) & nonnan for i in instance_ids
            ])

            keep = masks.sum(axis=(1, 2)) > 0
            instance_ids = instance_ids[keep]
            class_ids = class_ids[keep]
            Ts_cad2cam = Ts_cad2cam[keep]
            masks = masks[keep]

            pitch = np.array([
                self._models.get_voxel_pitch(self._voxel_dim, class_id=i)
                for i in class_ids
            ])
            centroids = np.array([np.mean(pcd_2s[m], axis=0) for m in masks])
            aabb_min = centroids - (self._voxel_dim / 2 - 0.5) * pitch[:, None]
            aabb_max = aabb_min + self._voxel_dim * pitch[:, None]

            for i, ins_id in enumerate(instance_ids):
                mapping.initialize(ins_id, pitch=pitch[i])
                mapping._octrees[ins_id].setBBXMin(aabb_min[i])
                mapping._octrees[ins_id].setBBXMax(aabb_max[i])
                mapping.integrate(ins_id, masks[i], pcd_2s)

            mapping.initialize(0, pitch=0.01)
            for ins_id in np.unique(instance_label_4s):
                if ins_id in instance_ids:
                    continue
                mapping._octrees[0].setBBXMin(aabb_min.min(axis=0))
                mapping._octrees[0].setBBXMax(aabb_max.max(axis=0))
                mapping.integrate(0, instance_label_4s == ins_id, pcd_4s)

        examples = []
        for instance_id, class_id, T_cad2cam in zip(
            instance_ids, class_ids, Ts_cad2cam
        ):
            if filter_class_ids and self._class_ids and \
                    class_id not in self._class_ids:
                continue

            mask = instance_label == instance_id
            if mask.sum() < self._mask_size_minimal:
                examples.append(self._get_invalid_data())
                continue

            bbox = objslampp.geometry.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
                examples.append(self._get_invalid_data())
                continue

            # augment
            if self._augmentation:
                rgb, depth, mask = self._augment(rgb, depth, mask)

            rgb = frame['rgb'].copy()
            rgb[~mask] = 0
            rgb = rgb[y1:y2, x1:x2]
            rgb = imgviz.centerize(rgb, (256, 256))

            pcd_ins = pcd.copy()
            pcd_ins[~mask] = np.nan
            pcd_ins = pcd_ins[y1:y2, x1:x2]
            pcd_ins = imgviz.centerize(pcd_ins, (256, 256), cval=np.nan)

            nonnan = ~np.isnan(pcd_ins).any(axis=2)
            if nonnan.sum() == 0:
                examples.append(self._get_invalid_data())
                continue

            pitch = self._get_pitch(class_id=class_id)
            centroid = np.nanmean(pcd_ins, axis=(0, 1))
            center = centroid + (0, 0, pitch * self._voxel_dim / 5)
            origin = center - pitch * (self._voxel_dim / 2. - 0.5)

            quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
            translation_true = tf.translation_from_matrix(T_cad2cam)

            example = dict(
                class_id=class_id,
                pitch=pitch,
                origin=origin,
                rgb=rgb,
                pcd=pcd_ins,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
            )

            if self._return_occupancy_grids:
                grid_target, grid_nontarget, grid_empty = \
                    mapping.get_target_grids(
                        instance_id,
                        dimensions=(self._voxel_dim,) * 3,
                        pitch=pitch,
                        origin=origin,
                    )
                example['grid_target'] = grid_target
                example['grid_nontarget'] = grid_nontarget
                example['grid_empty'] = grid_empty

            examples.append(example)
        return examples

    def get_example(self, index):
        examples = self.get_examples(index)

        class_ids = np.array([e['class_id'] for e in examples], dtype=int)

        if self._class_ids:
            keep = np.isin(class_ids, self._class_ids)
        else:
            # None or []
            keep = class_ids != -1

        if keep.sum():
            return [example for kp, example in zip(keep, examples) if kp]
        else:
            return [self._get_invalid_data()]

    def _augment(self, rgb, depth, mask):
        augmentation_all = {'rgb', 'depth'}
        assert augmentation_all.issuperset(set(self._augmentation))

        if 'rgb' in self._augmentation:
            rgb = self._augment_rgb(rgb)

        if 'depth' in self._augmentation:
            depth = self._augment_depth(depth)

        return rgb, depth, mask

    def _augment_rgb(self, rgb):
        augmenter = iaa.Sequential([
            iaa.ContrastNormalization(alpha=(0.8, 1.2)),
            iaa.WithColorspace(
                to_colorspace='HSV',
                from_colorspace='RGB',
                children=iaa.Sequential([
                    iaa.WithChannels(
                        (1, 2),
                        iaa.Multiply(mul=(0.8, 1.2), per_channel=True),
                    ),
                    iaa.WithChannels(
                        (0,),
                        iaa.Multiply(mul=(0.95, 1.05), per_channel=True),
                    ),
                ]),
            ),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.KeepSizeByResize(children=iaa.Resize((0.25, 1.0))),
        ])
        rgb = augmenter.augment_image(rgb)

        return rgb

    def _augment_depth(self, depth):
        depth = depth.copy()
        random_state = imgaug.current_random_state()
        depth += random_state.normal(scale=0.01, size=depth.shape)
        return depth
