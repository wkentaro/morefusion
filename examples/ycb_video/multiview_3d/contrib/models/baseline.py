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
        pitch,
        origin,
        values,
        points,
        quaternion_true=None,
        translation_true=None,
    ):
        xp = self.xp

        B = class_id.shape[0]
        dimensions = (self._voxel_dim,) * 3

        # prepare
        pitch = pitch.astype(np.float32)
        origin = origin.astype(np.float32)
        if quaternion_true is not None:
            quaternion_true = quaternion_true.astype(np.float32)
        if translation_true is not None:
            translation_true = translation_true.astype(np.float32)

        h = []
        actives = []
        for i in range(B):
            h_i, counts_i = objslampp.functions.average_voxelization_3d(
                values=values[i],
                points=points[i],
                origin=origin[i],
                pitch=pitch[i],
                dimensions=dimensions,
                channels=values[i].shape[1],
                return_counts=True,
            )  # CXYZ
            actives_i = counts_i[0] > 0

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
                for j in indices:
                    points_j = points[j]
                    T_cad2cam_j = objslampp.functions.transformation_matrix(
                        quaternion_true[j], translation_true[j]
                    )
                    points_j = objslampp.functions.transform_points(
                        points_j,
                        F.matmul(T_cad2cam_i, F.inv(T_cad2cam_j)),
                    )

                    h_j, counts_j = objslampp.functions.average_voxelization_3d(  # NOQA
                        values=values[j],
                        points=points_j,
                        origin=origin[i],
                        pitch=pitch[i],
                        dimensions=dimensions,
                        channels=values[j].shape[1],
                        return_counts=True,
                    )  # CXYZ

                    h_i = F.maximum(h_i, h_j)
                    actives_i = actives_i | (counts_j[0] > 0)

            h.append(h_i[None])
            actives.append(actives_i[None])

        h = F.concat(h, axis=0)           # BCXYZ
        actives = xp.concatenate(actives, axis=0)  # BXYZ

        return h, actives

    def predict(
        self,
        *,
        class_id,
        pitch,
        origin,
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
            quaternion_true = []
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
                quaternion_true_i = tf.quaternion_from_matrix(
                    cuda.to_cpu(T_cad2cam_true_i)
                )
                quaternion_true_i = xp.asarray(quaternion_true_i)
                quaternion_true.append(quaternion_true_i)
            quaternion_true = xp.stack(quaternion_true)

        h, actives = self._voxelize(
            class_id=class_id,
            pitch=pitch,
            origin=origin,
            values=values,
            points=points,
            quaternion_true=quaternion_true,
            translation_true=translation_true,
        )

        quaternion_pred, translation_pred = self._predict_from_voxel(
            class_id=class_id,
            pitch=pitch,
            origin=origin,
            matrix=h,
            actives=actives,
        )

        if chainer.config.train:
            return (
                quaternion_pred,
                translation_pred,
                quaternion_true,
                translation_true,
            )
        else:
            return quaternion_pred, translation_pred

    def __call__(
        self,
        *,
        class_id,
        pitch,
        origin,
        rgb,
        pcd,
        quaternion_true,
        translation_true,
    ):
        keep = class_id != -1
        if keep.sum() == 0:
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        class_id = class_id[keep]
        pitch = pitch[keep]
        origin = origin[keep]
        rgb = rgb[keep]
        pcd = pcd[keep]
        quaternion_true = quaternion_true[keep]
        translation_true = translation_true[keep]

        quaternion_pred, translation_pred, quaternion_true, translation_true =\
            self.predict(
                class_id=class_id,
                pitch=pitch,
                origin=origin,
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
