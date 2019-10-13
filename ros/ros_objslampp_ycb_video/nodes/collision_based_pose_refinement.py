#!/usr/bin/env python

import time

import chainer
from chainer.backends import cuda
import numpy as np

import objslampp

import message_filters
from ros_objslampp_msgs.msg import ObjectPoseArray
from ros_objslampp_msgs.msg import VoxelGridArray
import rospy
import topic_tools


class CollisionBasedPoseRefinement(topic_tools.LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        self._pcd_cad = {}

        super().__init__()
        self._pub = self.advertise('~output', ObjectPoseArray, queue_size=1)
        self._post_init()

    def subscribe(self):
        sub_pose = message_filters.Subscriber(
            '~input/poses', ObjectPoseArray, queue_size=1
        )
        sub_grid = message_filters.Subscriber(
            '~input/grids', VoxelGridArray, queue_size=1
        )
        sub_grid_noentry = message_filters.Subscriber(
            '~input/grids_noentry', VoxelGridArray, queue_size=1
        )
        self._subscribers = [sub_pose, sub_grid, sub_grid_noentry]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    @staticmethod
    def _grid_msg_to_matrix(grid):
        pitch = grid.pitch
        origin = np.array(
            [grid.origin.x, grid.origin.y, grid.origin.z],
            dtype=np.float32,
        )
        indices = np.array(grid.indices)
        k = indices % grid.dims.z
        j = indices // grid.dims.z % grid.dims.y
        i = indices // grid.dims.z // grid.dims.y
        dims = (grid.dims.x, grid.dims.y, grid.dims.z)
        matrix = np.zeros(dims, dtype=np.float32)
        matrix[i, j, k] = 1
        return matrix, pitch, origin

    def _get_pcd_cad(self, class_id):
        if class_id in self._pcd_cad:
            pcd_cad = self._pcd_cad[class_id]
        else:
            pitch = self._models.get_voxel_pitch(32, class_id)
            pcd_cad = self._models.get_solid_voxel(class_id).points
            pcd_cad = objslampp.extra.open3d.voxel_down_sample(
                pcd_cad, voxel_size=pitch
            ).astype(np.float32)
            pcd_cad = cuda.to_gpu(pcd_cad)
            self._pcd_cad[class_id] = pcd_cad
        return pcd_cad

    def _callback(self, poses_msg, grids_msg, grids_noentry_msg):
        grids = {
            g.instance_id: self._grid_msg_to_matrix(g)
            for g in grids_msg.grids
        }
        grids_noentry = {
            g.instance_id: self._grid_msg_to_matrix(g)
            for g in grids_noentry_msg.grids
        }

        points = []
        pitches = []
        origins = []
        grid_target = []
        grid_nontarget_empty = []
        transforms = []
        for i, pose in enumerate(poses_msg.poses):
            grid, pitch, origin = grids[pose.instance_id]
            grid_no, pitch_no, origin_no = grids_noentry[pose.instance_id]
            assert pitch == pitch_no
            assert (origin == origin_no).all()
            del pitch_no, origin_no
            grid_no[grid == 1] = 0

            translation = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ], dtype=np.float32)
            quaternion = np.array([
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
            ], dtype=np.float32)
            transform = objslampp.functions.transformation_matrix(
                quaternion, translation
            ).array

            points.append(self._get_pcd_cad(pose.class_id))
            pitches.append(cuda.to_gpu(np.float32(pitch)))
            origins.append(cuda.to_gpu(origin.astype(np.float32)))
            grid_target.append(cuda.to_gpu(grid))
            grid_nontarget_empty.append(cuda.to_gpu(grid_no))
            transforms.append(transform)
        grid_target = cuda.cupy.stack(grid_target)

        link = objslampp.contrib.CollisionBasedPoseRefinementLink(transforms)
        link.to_gpu()
        optimizer = chainer.optimizers.Adam(alpha=0.01)
        optimizer.setup(link)
        link.translation.update_rule.hyperparam.alpha *= 0.1

        t_start = time.time()
        for iteration in range(200):
            loss = link(
                points,
                pitches,
                origins,
                grid_target,
                grid_nontarget_empty,
            )
            loss.backward()
            optimizer.update()
            link.zerograds()

            if iteration % 10 == 0:
                rospy.loginfo(f'[{iteration}] {time.time() - t_start} [s]')

        quaternion = cuda.to_cpu(link.quaternion.array)
        translation = cuda.to_cpu(link.translation.array)
        for i, pose in enumerate(poses_msg.poses):
            pose.pose.position.x = translation[i][0]
            pose.pose.position.y = translation[i][1]
            pose.pose.position.z = translation[i][2]
            pose.pose.orientation.w = quaternion[i][0]
            pose.pose.orientation.x = quaternion[i][1]
            pose.pose.orientation.y = quaternion[i][2]
            pose.pose.orientation.z = quaternion[i][3]
        self._pub.publish(poses_msg)


if __name__ == '__main__':
    rospy.init_node('collision_based_pose_refinement')
    CollisionBasedPoseRefinement()
    rospy.spin()
