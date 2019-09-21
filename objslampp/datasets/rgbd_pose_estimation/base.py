import imgviz
import numpy as np
import trimesh
import trimesh.transformations as tf

from ..base import DatasetBase
from ... import geometry as geometry_module
from ...contrib import MultiInstanceOctreeMapping


class RGBDPoseEstimationDatasetBase(DatasetBase):

    _n_points_minimal = 1
    _image_size = 256
    _voxel_dim = 32

    def __init__(
        self,
        models,
        class_ids=None,
    ):
        self._models = models
        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids
        self._random_state = np.random.mtrand._rand

    def get_frame(self, index):
        raise NotImplementedError

    def build_octomap(self, pcd, instance_label, instance_ids, class_ids):
        mapping = MultiInstanceOctreeMapping()
        nonnan = ~np.isnan(pcd).any(axis=2)

        # map foreground objects
        for instance_id, class_id in zip(instance_ids, class_ids):
            if class_id <= 0:
                continue
            mask = (instance_label == instance_id) & nonnan
            pitch = self._models.get_voxel_pitch(self._voxel_dim, class_id)
            mapping.initialize(instance_id, pitch=pitch)
            mapping.integrate(instance_id, mask, pcd)

        # map background objects
        mapping.initialize(0, pitch=0.01)
        for instance_id in np.unique(instance_label):
            if instance_id in instance_ids:  # foreground
                continue
            mask = (instance_label == instance_id) & nonnan
            mapping.integrate(0, mask, pcd)
        return mapping

    def _get_grid_full(self, examples, pitch, origin):
        dims = (self._voxel_dim,) * 3
        grid_full = np.zeros(dims, dtype=np.int32)
        for i, example in enumerate(examples):
            T = tf.quaternion_matrix(example['quaternion_true'])
            T = geometry_module.compose_transform(
                R=T[:3, :3], t=example['translation_true']
            )
            vox = self._models.get_solid_voxel(example['class_id'])
            points = trimesh.transform_points(vox.points, T)
            indices = trimesh.voxel.points_to_indices(
                points, pitch=pitch, origin=origin
            )
            I, J, K = indices[:, 0], indices[:, 1], indices[:, 2]
            keep = (
                (0 <= I) & (I < dims[0]) &
                (0 <= J) & (J < dims[1]) &
                (0 <= K) & (K < dims[2])
            )
            I, J, K = I[keep], J[keep], K[keep]
            grid_full[I, J, K] = i + 1  # starts from 1
        return grid_full

    def get_example(self, index):
        frame = self.get_frame(index)

        instance_ids = frame['instance_ids']
        class_ids = frame['class_ids']
        rgb = frame['rgb']
        depth = frame['depth']
        instance_label = frame['instance_label']
        K = frame['intrinsic_matrix']
        Ts_cad2cam = frame['Ts_cad2cam']
        pcd = geometry_module.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        )

        if instance_ids.size == 0:
            return []

        mapping = self.build_octomap(
            pcd, instance_label, instance_ids, class_ids
        )

        examples = []
        for instance_id, class_id, T_cad2cam in zip(
            instance_ids, class_ids, Ts_cad2cam
        ):
            if class_id == 0:
                continue

            if self._class_ids and class_id not in self._class_ids:
                continue

            mask = instance_label == instance_id
            bbox = geometry_module.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
                continue

            pcd_ins = pcd.copy()
            pcd_ins[~mask] = np.nan
            pcd_ins = pcd_ins[y1:y2, x1:x2]
            nonnan = ~np.isnan(pcd_ins).any(axis=2)
            if nonnan.sum() < self._n_points_minimal:
                continue
            pcd_ins = imgviz.centerize(
                pcd_ins,
                (self._image_size, self._image_size),
                cval=np.nan,
                interpolation='nearest',
            )

            rgb_ins = rgb.copy()
            rgb_ins[~mask] = 0
            rgb_ins = rgb_ins[y1:y2, x1:x2]
            rgb_ins = imgviz.centerize(
                rgb_ins, (self._image_size, self._image_size)
            )

            quaternion_true = tf.quaternion_from_matrix(T_cad2cam)
            translation_true = tf.translation_from_matrix(T_cad2cam)

            center = np.nanmedian(pcd_ins, axis=(0, 1))
            dim = self._voxel_dim
            pitch = self._models.get_voxel_pitch(self._voxel_dim, class_id)
            origin = center - (dim / 2 - 0.5) * pitch
            grid_target, grid_nontarget, grid_empty = mapping.get_target_grids(
                instance_id,
                dimensions=(dim, dim, dim),
                pitch=pitch,
                origin=origin,
            )

            example = dict(
                class_id=class_id,
                rgb=rgb_ins,
                pcd=pcd_ins,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
                origin=origin,
                pitch=pitch,
                grid_target=grid_target,
                grid_nontarget=grid_nontarget,
                grid_empty=grid_empty,
            )

            examples.append(example)

        n_examples = len(examples)
        for i_target, example in enumerate(examples):
            assert example['class_id'] >= 1

            indices = np.arange(n_examples)
            indices_nontarget = indices[indices != i_target]
            examples_nontarget = [examples[i] for i in indices_nontarget]

            pitch = example['pitch']
            origin = example['origin']
            grid_target_full = self._get_grid_full(
                [example], pitch, origin
            )
            grid_nontarget_full = self._get_grid_full(
                examples_nontarget, pitch, origin
            )
            example['grid_target_full'] = grid_target_full
            example['grid_nontarget_full'] = grid_nontarget_full

        return examples
