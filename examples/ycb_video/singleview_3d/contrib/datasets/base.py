import chainer
import imgaug
import imgaug.augmenters as iaa
import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp

from .multi_instance_octree_mapping import MultiInstanceOctreeMapping


class DatasetBase(objslampp.datasets.DatasetBase):

    _models = objslampp.datasets.YCBVideoModels()
    voxel_dim = 32

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

    def _get_invalid_data(self):
        return dict(
            class_id=-1,
            pitch=0.,
            origin=np.zeros((3,), dtype=np.float64),
            rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            pcd=np.zeros((256, 256, 3), dtype=np.float64),
            quaternion_true=np.zeros((4,), dtype=np.float64),
            translation_true=np.zeros((3,), dtype=np.float64),
        )

    def _get_pitch(self, class_id):
        return self._models.get_voxel_pitch(
            dimension=self.voxel_dim, class_id=class_id
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
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        if chainer.is_debug():
            print(f'[{index:08d}]: class_ids: {class_ids.tolist()}')
            print(f'[{index:08d}]: instance_ids: {instance_ids.tolist()}')

        if self._return_occupancy_grids:
            mapping = MultiInstanceOctreeMapping()
            mapping.initialize(0, pitch=0.01)
            mapping.integrate(0, instance_label == 0, pcd)
            for instance_id, class_id in zip(instance_ids, class_ids):
                pitch = self._get_pitch(class_id)
                mapping.initialize(instance_id, pitch=pitch)
                mapping.integrate(
                    instance_id, instance_label == instance_id, pcd
                )

        examples = []
        for instance_id, class_id, T_cad2cam in zip(
            instance_ids, class_ids, Ts_cad2cam
        ):
            if self._class_ids and class_id not in self._class_ids:
                continue

            mask = instance_label == instance_id
            if mask.sum() == 0:
                continue

            bbox = objslampp.geometry.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
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
                continue

            pitch = self._get_pitch(class_id=class_id)
            centroid = np.nanmean(pcd_ins, axis=(0, 1))
            origin = centroid - pitch * (self.voxel_dim / 2. - 0.5)

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
                        dimensions=(self.voxel_dim,) * 3,
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

        class_ids = [e['class_id'] for e in examples]

        if self._class_ids:
            options = set(self._class_ids) & set(class_ids)
            if options:
                class_id = np.random.choice(list(options))
            else:
                return self._get_invalid_data()
        else:
            # None or []
            class_id = np.random.choice(class_ids)
        instance_index = np.random.choice(np.where(class_ids == class_id)[0])

        return examples[instance_index]

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
