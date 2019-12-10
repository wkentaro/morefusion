#!/usr/bin/env python

import copy
import itertools
import queue

import trimesh.transformations as ttf

import numpy as np
import morefusion

from morefusion_panda_ycb_video.msg import ObjectClassArray
from morefusion_panda_ycb_video.msg import ObjectPose
from morefusion_panda_ycb_video.msg import ObjectPoseArray
from morefusion_panda_ycb_video.msg import VoxelGridArray
import rospy
import tf


class Object:

    _n_votes = 3
    _add_threshold = 0.02
    _adds_threshold = 0.01

    def __init__(self, class_id, pcd, is_symmetric):
        self.class_id = class_id
        self._pcd = pcd
        self._is_symmetric = is_symmetric

        self._poses = queue.deque([], 6)
        self.is_spawned = False

    @property
    def pose(self):
        if not self.is_spawned:
            return
        return self._poses[-1]

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'class_id={self.class_id}, '
            f'n_poses={len(self._poses)}, '
            f'is_spawned={self.is_spawned})'
        )

    def append_pose(self, pose):
        if not self.is_spawned:
            self._poses.append(pose)

    def validate(self):
        if self.is_spawned:
            return True  # already validated before

        if len(self._poses) < (self._n_votes):
            return False  # too early to decide

        latest_pose = self._poses[-1]
        poses = np.array(
            list(itertools.islice(self._poses, len(self._poses) - 1))
        )
        add, add_s = morefusion.metrics.average_distance(
            [self._pcd] * len(poses),
            [latest_pose] * len(poses),
            poses,
        )
        if self._is_symmetric:
            add = add_s
        del add_s
        add = np.array(add)
        # rospy.logerr('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # rospy.logerr(f'{self.class_id}: {add}')
        # rospy.logerr('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if self._is_symmetric:
            threshold = self._adds_threshold
        else:
            threshold = self._add_threshold
        if (add < threshold).sum() >= (self._n_votes - 1):
            self.is_spawned = True
            self._poses = tuple(self._poses)  # freeze it
            return True
        return False


class ObjectMapping:

    _models = morefusion.datasets.YCBVideoModels()

    def __init__(self):
        self._objects = {}  # instance_id: Object()
        self._instance_ids_removed = set()
        self._base_frame = rospy.get_param('~frame_id', 'map')
        self._pub = rospy.Publisher(
            '~output/poses', ObjectPoseArray, queue_size=1, latch=True
        )
        self._pub_grids = rospy.Publisher(
            '~output/grids', VoxelGridArray, queue_size=1, latch=True
        )

        self._tf_listener = tf.TransformListener(cache_time=rospy.Duration(30))
        self._sub = rospy.Subscriber(
            '~input/poses', ObjectPoseArray, self._callback, queue_size=1
        )
        self._sub_grids = rospy.Subscriber(
            '~input/grids', VoxelGridArray, self._callback_grids, queue_size=1
        )
        self._sub_remove = rospy.Subscriber(
            '~input/remove',
            ObjectClassArray,
            self._callback_remove,
            queue_size=1,
        )

    def _callback_grids(self, grids_msg):
        out_msg = copy.deepcopy(grids_msg)
        out_msg.grids = []
        for grid in grids_msg.grids:
            if (grid.instance_id in self._objects and
                    self._objects[grid.instance_id].is_spawned):
                continue
            out_msg.grids.append(grid)
        self._pub_grids.publish(out_msg)

    def _callback_remove(self, cls_msg):
        for cls in cls_msg.classes:
            self._instance_ids_removed.add(cls.instance_id)
        self._publish_poses(cls_msg.header.stamp)

    def _publish_poses(self, stamp):
        out_msg = ObjectPoseArray()
        out_msg.header.stamp = stamp
        out_msg.header.frame_id = self._base_frame
        for ins_id, obj in self._objects.items():
            if ins_id in self._instance_ids_removed:
                continue
            if not obj.validate():
                continue

            pose = ObjectPose(
                instance_id=ins_id,
                class_id=obj.class_id,
            )
            T_cad2base = obj.pose
            translation = ttf.translation_from_matrix(T_cad2base)
            quaternion = ttf.quaternion_from_matrix(T_cad2base)
            pose.pose.position.x = translation[0]
            pose.pose.position.y = translation[1]
            pose.pose.position.z = translation[2]
            pose.pose.orientation.w = quaternion[0]
            pose.pose.orientation.x = quaternion[1]
            pose.pose.orientation.y = quaternion[2]
            pose.pose.orientation.z = quaternion[3]
            out_msg.poses.append(pose)
        self._pub.publish(out_msg)

    def _callback(self, poses_msg):
        try:
            self._tf_listener.waitForTransform(
                target_frame=self._base_frame,
                source_frame=poses_msg.header.frame_id,
                time=poses_msg.header.stamp,
                timeout=rospy.Duration(0.1),
            )
        except Exception as e:
            rospy.logerr(e)
            return

        translation, quaternion = self._tf_listener.lookupTransform(
            target_frame=self._base_frame,
            source_frame=poses_msg.header.frame_id,
            time=poses_msg.header.stamp,
        )
        translation = np.asarray(translation)
        quaternion = np.asarray(quaternion)[[3, 0, 1, 2]]
        T_cam2base = morefusion.functions.transformation_matrix(
            quaternion, translation
        ).array

        # ---------------------------------------------------------------------

        for pose in poses_msg.poses:
            instance_id = pose.instance_id
            class_id = pose.class_id
            quaternion, translation = morefusion.ros.from_ros_pose(pose.pose)
            T_cad2cam = morefusion.functions.transformation_matrix(
                quaternion, translation
            ).array
            T_cad2base = T_cam2base @ T_cad2cam

            if instance_id in self._objects:
                self._objects[instance_id].append_pose(T_cad2base)
            else:
                self._objects[instance_id] = Object(
                    class_id=class_id,
                    pcd=self._models.get_pcd(class_id),
                    is_symmetric=class_id in
                    morefusion.datasets.ycb_video.class_ids_symmetric
                )
                self._objects[instance_id].append_pose(T_cad2base)

        self._publish_poses(stamp=poses_msg.header.stamp)


if __name__ == '__main__':
    rospy.init_node('object_mapping', disable_signals=True)
    ObjectMapping()
    rospy.spin()
