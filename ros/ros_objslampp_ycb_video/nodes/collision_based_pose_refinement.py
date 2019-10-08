#!/usr/bin/env python
# flake8: noqa

import numpy as np
import trimesh.transformations as ttf

import objslampp

import message_filters
from ros_objslampp_msgs.msg import ObjectPoseArray
from ros_objslampp_msgs.msg import VoxelGridArray
import rospy
import topic_tools


class CollisionBasedPoseRefinement(topic_tools.LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', ObjectPoseArray, queue_size=1)
        self._post_init()
        self.subscribe()

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

    def _callback(self, poses_msg, grids_msg, grids_noentry_msg):
        grids = {g.instance_id: g for g in grids_msg.grids}
        grids_noentry = {g.instance_id: g for g in grids_noentry_msg.grids}

        for i, pose in enumerate(poses_msg.poses):
            grid_target, pitch1, origin1 = self._grid_msg_to_matrix(
                grids[pose.instance_id]
            )
            grid_nontarget_empty, pitch2, origin2 = self._grid_msg_to_matrix(
                grids_noentry[pose.instance_id]
            )
            assert pitch1 == pitch2
            assert (origin1 == origin2).all()
            pitch = pitch1
            origin = origin1

            pcd_cad = self._models.get_solid_voxel(
                pose.class_id
            ).points.astype(np.float32)
            pcd_cad = objslampp.extra.open3d.voxel_down_sample(
                pcd_cad, voxel_size=pitch
            ).astype(np.float32)
            translation = np.array(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
                dtype=np.float32,
            )
            quaternion = np.array([
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
            ], dtype=np.float32)
            transform_init = objslampp.functions.transformation_matrix(
                quaternion, translation
            ).array

            '''
            registration = objslampp.contrib.OccupancyRegistration(
                pcd_cad,
                np.stack((grid_target, grid_nontarget_empty)),
                pitch=pitch,
                origin=origin,
                threshold=2,
                transform_init=transform_init,  # cad2depth
                gpu=0,
                alpha=0.1
            )
            transform = registration.register(iteration=30)
            quaternion = ttf.quaternion_from_matrix(transform)
            translation = ttf.translation_from_matrix(transform)
            poses_msg.poses[i].pose.position.x = translation[0]
            poses_msg.poses[i].pose.position.y = translation[1]
            poses_msg.poses[i].pose.position.z = translation[2]
            poses_msg.poses[i].pose.orientation.w = quaternion[0]
            poses_msg.poses[i].pose.orientation.x = quaternion[1]
            poses_msg.poses[i].pose.orientation.y = quaternion[2]
            poses_msg.poses[i].pose.orientation.z = quaternion[3]
            break
            '''
        self._pub.publish(poses_msg)


if __name__ == '__main__':
    rospy.init_node('collision_based_pose_refinement')
    CollisionBasedPoseRefinement()
    rospy.spin()
