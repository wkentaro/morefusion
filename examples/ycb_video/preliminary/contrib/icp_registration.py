import chainer

import numpy as np
import open3d
import trimesh.transformations as tf


class ICPRegistration:

    def __init__(self, pcd_depth, pcd_cad, transform_init=None):
        self._pcd_depth = pcd_depth
        self._pcd_cad = pcd_cad

        if transform_init is None:
            transform_init = np.eye(4)
        self._transform = transform_init

    def _prepare(self, voxel_size):
        source = open3d.PointCloud()
        source.points = open3d.Vector3dVector(self._pcd_depth)

        target = open3d.PointCloud()
        target.points = open3d.Vector3dVector(self._pcd_cad)

        source = open3d.voxel_down_sample(source, voxel_size=voxel_size)
        target = open3d.voxel_down_sample(target, voxel_size=voxel_size)

        return source, target

    def register(self, iteration=None, voxel_size=None):
        iteration = 100 if iteration is None else iteration
        voxel_size = 0.01 if voxel_size is None else voxel_size

        source, target = self._prepare(voxel_size=voxel_size)
        result = open3d.registration_icp(
            source,  # points_from_depth
            target,  # points_from_cad
            2 * voxel_size,
            tf.inverse_matrix(self._transform),
            open3d.TransformationEstimationPointToPoint(False),
            open3d.ICPConvergenceCriteria(max_iteration=iteration),
        )
        return tf.inverse_matrix(result.transformation)

    def register_iterative(self, iteration=None, voxel_size=None):
        iteration = 100 if iteration is None else iteration
        voxel_size = 0.01 if voxel_size is None else voxel_size

        yield self._transform

        source, target = self._prepare(voxel_size=voxel_size)
        for i in range(iteration):
            result = open3d.registration_icp(
                source,  # points_from_depth
                target,  # points_from_cad
                2 * voxel_size,
                tf.inverse_matrix(self._transform),
                open3d.TransformationEstimationPointToPoint(False),
                open3d.ICPConvergenceCriteria(max_iteration=1),
            )
            print(f'[{i:08d}] fitness={result.fitness:.2g} '
                  f'inlier_rmse={result.inlier_rmse:.2g}')
            self._transform = tf.inverse_matrix(
                result.transformation
            )
            yield self._transform
