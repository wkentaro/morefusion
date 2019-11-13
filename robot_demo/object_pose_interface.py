import math
import numpy as np
from robot_demo import general_kinematics as gk
from geometry_msgs.msg import Pose


class ObjectPoseInterface:

    def __init__(self, object_models, axis_order):
        self._axis_order = axis_order
        self._floor_z = 0.07
        self._object_models = object_models
        self._class_names = self._object_models.class_names
        self._class_ids = range(1,len(self._object_models.class_names))

        # Define canonical models
        self._define_canonical_object_rotations()
        self._define_canonical_extents()

    # Init #
    # -----#

    def _define_canonical_object_rotations(self):

        self._canonical_quaternions = dict()

        for class_id in self._class_ids:

            self._canonical_quaternions[class_id] = dict()

            self._canonical_quaternions[class_id]['x+'] = \
            [gk.euler_to_quaternion(np, [0, math.pi / 2, -math.pi]),
             gk.euler_to_quaternion(np, [0, -math.pi/2, 0]),
             gk.euler_to_quaternion(np, [math.pi / 2, 0, math.pi / 2]),
             gk.euler_to_quaternion(np, [-math.pi / 2, 0, -math.pi / 2])]

            self._canonical_quaternions[class_id]['x-'] = list()
            self._canonical_quaternions[class_id]['y+'] = list()
            self._canonical_quaternions[class_id]['y-'] = list()
            self._canonical_quaternions[class_id]['z+'] = list()
            self._canonical_quaternions[class_id]['z-'] = list()

    def _define_canonical_extents(self):

        self._canonical_extents = dict()

        for class_id in self._class_ids:

            class_name = self._class_names[class_id]
            extents = self._object_models.get_cad(class_name=class_name).extents

            self._canonical_extents[class_id] = dict()

            self._canonical_extents[class_id]['x+'] = \
                [[extents[2], extents[1], extents[0]],
                 [extents[2], extents[1], extents[0]],
                 [extents[1], extents[2], extents[0]],
                 [extents[1], extents[2], extents[0]]]

            self._canonical_extents[class_id]['x-'] = list()

            self._canonical_extents[class_id]['y+'] = list()

            self._canonical_extents[class_id]['y-'] = list()

            self._canonical_extents[class_id]['z+'] = list()

            self._canonical_extents[class_id]['z-'] = list()

    def _get_object_poses(self, class_id, position_on_table):
        object_quaternions = list()
        object_extents = list()
        for axis in self._axis_order:
            object_quaternions += self._canonical_quaternions[class_id][axis]
            object_extents += self._canonical_extents[class_id][axis]
        object_z_positions = [self._floor_z + extent[2]/2 for extent in object_extents]
        object_positions = [list(position_on_table) + [z_position] for z_position in object_z_positions]
        return [list(object_position) + list(object_quaternion) for object_position, object_quaternion
                in zip(object_positions, object_quaternions)]

    def get_robot_poses(self, class_id, object_to_robot_mat, position_on_table):
        object_poses = self._get_object_poses(class_id, position_on_table)
        object_mats = [gk.quaternion_pose_to_mat(np, pose) for pose in object_poses]
        robot_mats = [np.matmul(object_mat, object_to_robot_mat) for object_mat in object_mats]
        robot_poses = [gk.mat_to_quaternion_pose(np, robot_mat) for robot_mat in robot_mats]

        robot_pose_msgs = list()
        for robot_pose in robot_poses:
            pose_msg = Pose()
            pose_msg.position.x = robot_pose[0]
            pose_msg.position.y = robot_pose[1]
            pose_msg.position.z = robot_pose[2]
            pose_msg.orientation.x = robot_pose[3]
            pose_msg.orientation.y = robot_pose[4]
            pose_msg.orientation.z = robot_pose[5]
            pose_msg.orientation.w = robot_pose[6]
            robot_pose_msgs.append(pose_msg)
        return robot_pose_msgs

'''
if __name__ == '__main__':
    import objslampp.datasets.ycb_video as ycb_video_dataset
    object_models = ycb_video_dataset.YCBVideoModels()

    opi = ObjectPoseInterface(object_models, ['x-','x+','z-','z+','y-','y+'])
    robot_to_object_mat = np.eye(4)
    robot_poses = opi.get_robot_poses(1, robot_to_object_mat, [0,0])
'''
