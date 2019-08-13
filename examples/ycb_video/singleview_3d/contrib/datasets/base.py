import imgviz
import numpy as np
import trimesh.transformations as tf

import objslampp


class DatasetBase(objslampp.datasets.DatasetBase):

    _models = objslampp.datasets.YCBVideoModels()
    _voxel_dim = 32
    _n_points_minimal = 50

    def __init__(
        self,
        root_dir=None,
        class_ids=None,
    ):
        self._root_dir = root_dir
        if class_ids is not None:
            class_ids = tuple(class_ids)
        self._class_ids = class_ids

    def _get_pitch(self, class_id):
        return self._models.get_voxel_pitch(
            dimension=self._voxel_dim, class_id=class_id
        )

    def get_example(self, index):
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
            return []

        examples = []
        for instance_id, class_id, T_cad2cam in zip(
            instance_ids, class_ids, Ts_cad2cam
        ):
            if self._class_ids and class_id not in self._class_ids:
                continue

            mask = instance_label == instance_id
            bbox = objslampp.geometry.masks_to_bboxes(mask)
            y1, x1, y2, x2 = bbox.round().astype(int)
            if (y2 - y1) * (x2 - x1) == 0:
                continue

            pcd_ins = pcd.copy()
            pcd_ins[~mask] = np.nan
            pcd_ins = pcd_ins[y1:y2, x1:x2]
            nonnan = ~np.isnan(pcd_ins).any(axis=2)
            if nonnan.sum() < self._n_points_minimal:
                continue
            pcd_ins = imgviz.centerize(pcd_ins, (256, 256), cval=np.nan)

            rgb_ins = rgb.copy()
            rgb_ins[~mask] = 0
            rgb_ins = rgb_ins[y1:y2, x1:x2]
            rgb_ins = imgviz.centerize(rgb_ins, (256, 256))

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
                rgb=rgb_ins,
                pcd=pcd_ins,
                quaternion_true=quaternion_true,
                translation_true=translation_true,
            )

            examples.append(example)
        return examples
