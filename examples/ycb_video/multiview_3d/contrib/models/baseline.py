import chainer
from chainer.backends import cuda
import chainer.functions as F
import numpy as np
import trimesh.transformations as tf

import objslampp

from .singleview_3d_baseline import BaselineModel as SingleView3DBaselineModel


class BaselineModel(SingleView3DBaselineModel):

    def _voxelize(
        self,
        class_id,
        values,
        points,
        quaternion_true=None,
        translation_true=None,
        pitch=None,
        origin=None,
    ):
        xp = self.xp

        B = class_id.shape[0]
        dim = self._voxel_dim
        dimensions = (dim, dim, dim)

        # prepare
        if quaternion_true is not None:
            quaternion_true = quaternion_true.astype(np.float32)
        if translation_true is not None:
            translation_true = translation_true.astype(np.float32)
        if pitch is None:
            pitch = xp.full((B,), np.nan, dtype=np.float32)
        if origin is None:
            origin = xp.full((B, 3), np.nan, dtype=np.float32)

        h = []
        count = []
        for i in range(B):
            if xp.isnan(pitch[i]):
                pitch[i] = self._models.get_voxel_pitch(
                    dimension=self._voxel_dim, class_id=int(class_id[i]),
                )
            if xp.isnan(origin[i]).any():
                center_i = objslampp.extra.cupy.median(points[i], axis=0)
                origin[i] = center_i - pitch[i] * (dim / 2. - 0.5)
                del center_i

            points_js = {}
            if chainer.config.train:
                T_cad2cam_i = objslampp.functions.transformation_matrix(
                    quaternion_true[i], translation_true[i]
                )
                indices = xp.where(class_id[i])[0]
                indices = indices[indices != i].tolist()
                if len(indices) > 0:
                    n_fuse = np.random.randint(0, len(indices) + 1)
                    if n_fuse > 0:
                        indices = np.random.choice(
                            indices, n_fuse, replace=False
                        )
                    del n_fuse
                origins = [origin[i]]
                for j in indices:
                    points_j = points[j]
                    T_cad2cam_j = objslampp.functions.transformation_matrix(
                        quaternion_true[j], translation_true[j]
                    ).array
                    points_j = objslampp.functions.transform_points(
                        points_j,
                        F.matmul(T_cad2cam_i, F.inv(T_cad2cam_j)),
                    ).array
                    center_j = objslampp.extra.cupy.median(points_j, axis=0)
                    origin_j = center_j - pitch[i] * (dim / 2. - 0.5)
                    points_js[j] = points_j
                    origins.append(origin_j)
                    del T_cad2cam_j, center_j, points_j
                origin[i] = xp.stack(origins).mean(axis=0)
                del T_cad2cam_i, indices, origins

            h_i, count_i = objslampp.functions.average_voxelization_3d(
                values=values[i],
                points=points[i],
                origin=origin[i],
                pitch=pitch[i],
                dimensions=dimensions,
                return_counts=True,
            )  # CXYZ
            if (count_i == 0).all():
                center_i = objslampp.extra.cupy.median(points[i], axis=0)
                origin[i] = center_i - pitch[i] * (dim / 2. - 0.5)
                del center_i
                h_i, count_i = objslampp.functions.average_voxelization_3d(
                    values=values[i],
                    points=points[i],
                    origin=origin[i],
                    pitch=pitch[i],
                    dimensions=dimensions,
                    return_counts=True,
                )  # CXYZ

            for j, points_j in points_js.items():
                h_j, count_j = objslampp.functions.average_voxelization_3d(
                    values=values[j],
                    points=points_j,
                    origin=origin[i],
                    pitch=pitch[i],
                    dimensions=dimensions,
                    return_counts=True,
                )  # CXYZ
                count_i = count_i + count_j
                denominator = count_i.copy()
                denominator[denominator == 0] = 1
                h_i = (h_i * count_i + h_j * count_j) / denominator
                del j, h_j, count_j, denominator
            del points_js

            h.append(h_i)
            count.append(count_i)
            del h_i, count_i

        h = F.stack(h)           # BCXYZ
        count = xp.stack(count)           # BCXYZ

        return pitch, origin, h, count

    def predict(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true=None,
        translation_true=None,
    ):
        xp = self.xp
        B = class_id.shape[0]

        values, points = self._extract(
            rgb=rgb,
            pcd=pcd,
        )

        if chainer.config.train:
            assert quaternion_true is not None
            assert translation_true is not None
            quaternion_true = quaternion_true.astype(np.float32)
            T_cad2cam_true = objslampp.functions.transformation_matrix(
                quaternion_true, translation_true
            ).array
            for i in range(B):
                T_cam2cad_true_i = F.inv(T_cad2cam_true[i]).array
                points[i] = objslampp.functions.transform_points(
                    points[i], T_cam2cad_true_i
                ).array
                T_random_rot = xp.asarray(
                    tf.random_rotation_matrix(), dtype=np.float32
                )
                T_cad2cam_true_i = T_cad2cam_true[i] @ T_random_rot
                points[i] = objslampp.functions.transform_points(
                    points[i], T_cad2cam_true_i
                ).array
                quaternion_true[i] = xp.asarray(tf.quaternion_from_matrix(
                    cuda.to_cpu(T_cad2cam_true_i)
                ))

        pitch, origin, voxelized, count = self._voxelize(
            class_id=class_id,
            values=values,
            points=points,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )

        quaternion_pred, translation_pred = self._predict_from_voxelized(
            class_id=class_id,
            pitch=pitch,
            origin=origin,
            voxelized=voxelized,
            count=count,
        )

        return quaternion_pred, translation_pred

    def __call__(
        self,
        *,
        class_id,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        quaternion_pred, translation_pred = self.predict(
            class_id=class_id,
            rgb=rgb,
            pcd=pcd,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )

        self.evaluate(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
        )

        loss = self.loss(
            class_id=class_id,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
            quaternion_pred=quaternion_pred,
            translation_pred=translation_pred,
        )
        return loss
