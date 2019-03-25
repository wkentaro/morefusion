import chainer.functions as F
import numpy as np
import trimesh.transformations as tf

from ..functions import compose_transform
from ..functions import quaternion_matrix


class Camera2CAD(object):

    def __init__(self, voxel_dim, pitch, origin_cad, origin_scan):
        assert origin_cad.shape == (3,)
        assert origin_scan.shape == (3,)
        self._voxel_dim = voxel_dim
        self._pitch = pitch
        self._origin_scan = origin_scan
        self._origin_cad = origin_cad

    def _translation_metric2voxel(self, x):
        return x / self._pitch / self._voxel_dim

    def _translation_voxel2metric(self, x):
        return x * self._pitch * self._voxel_dim

    def matrix_from_target_func(self, quaternion, translation):
        R_cam2cad = quaternion_matrix(quaternion[None])[0]
        T_cad2cam = F.inv(R_cam2cad)

        translation = self._translation_voxel2metric(translation)
        translation_concat = self._origin_cad - self._origin_scan
        translation = - (translation + translation_concat)
        T_cad2cam = compose_transform(
            T_cad2cam[None, :3, :3], translation[None]
        )[0]

        T_cam2cad = F.inv(T_cad2cam)
        return T_cam2cad

    def matrix_from_target(self, quaternion_target, translation_target):
        R_cam2cad = tf.quaternion_matrix(quaternion_target)
        T_cad2cam = tf.inverse_matrix(R_cam2cad)

        translation = self._translation_voxel2metric(translation_target)
        translation_concat = self._origin_cad - self._origin_scan
        T_cad2cam[:3, 3] = - (translation + translation_concat)

        T_cam2cad = tf.inverse_matrix(T_cad2cam)
        return T_cam2cad

    def target_from_matrix(self, T_cam2cad):
        T_cad2cam = tf.inverse_matrix(T_cam2cad)

        quaternion_target = tf.quaternion_from_matrix(T_cam2cad)
        quaternion_target = quaternion_target.astype(np.float32)

        translation_concat = self._origin_cad - self._origin_scan
        translation_target = \
            - tf.translation_from_matrix(T_cad2cam) - translation_concat
        translation_target = self._translation_metric2voxel(translation_target)
        translation_target = translation_target.astype(np.float32)

        return quaternion_target, translation_target
