import math
import numpy as np
from robot_demo import general_kinematics as gk
import trimesh
import trimesh.transformations as ttf
from geometry_msgs.msg import Pose


class ObjectPoseInterface:

    def __init__(self, object_models):
        self._floor_z = 0.07
        self._object_models = object_models
        self._class_names = self._object_models.class_names
        self._class_ids = range(1,len(self._object_models.class_names))

        # define class specific up axes
        # ToDo: define these for all object classes
        self._class_up_axis_keys = dict()
        self._class_up_axis_keys[1] = ['x+']

        # Define canonical models
        canonical_quaternions, canonical_extents = self._get_cannonical_object_rotations_and_extents()
        self._get_class_specific_orientations_and_extents(canonical_quaternions, canonical_extents)

    # Init #
    # -----#

    @staticmethod
    def _get_remaining_axis_key(axis_1, axis_2):
        if (axis_1[0] == 'y' and axis_2[0] == 'z') or (axis_2[0] == 'y' and axis_1[0] == 'z'):
            return 'x+'
        elif (axis_1[0] == 'x' and axis_2[0] == 'z') or (axis_2[0] == 'x' and axis_1[0] == 'z'):
            return 'y+'
        elif (axis_1[0] == 'x' and axis_2[0] == 'y') or (axis_2[0] == 'x' and axis_1[0] == 'y'):
            return 'z+'

    def _get_cannonical_object_rotations_and_extents(self):

        # final data container
        all_q_s = dict()
        all_extents = dict()

        # visualization
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.axis())

        # axes
        axes = dict()
        axes['x+'] = [1., 0., 0.]
        axes['x-'] = [-1., 0., 0.]
        axes['y+'] = [0., 1., 0.]
        axes['y-'] = [0., -1., 0.]
        axes['z+'] = [0., 0., 1.]
        axes['z-'] = [0., 0., -1.]

        # axis indices
        axis_indices = dict()
        axis_indices['x+'] = 0
        axis_indices['x-'] = 0
        axis_indices['y+'] = 1
        axis_indices['y-'] = 1
        axis_indices['z+'] = 2
        axis_indices['z-'] = 2

        # axis up transformations
        transformations = dict()
        transformations['x+'] = ('y+', -math.pi / 2)
        transformations['x-'] = ('y+', math.pi / 2)
        transformations['y+'] = ('x+', math.pi / 2)
        transformations['y-'] = ('x+', -math.pi / 2)
        transformations['z+'] = ('x+', 0.)
        transformations['z-'] = ('x+', math.pi)

        for up_axis_key, transformation in transformations.items():

            rotation_axis_key = transformation[0]
            remaining_axis_key = self._get_remaining_axis_key(up_axis_key, rotation_axis_key)
            rotation_angle = transformation[1]
            rotation_axis = axes[rotation_axis_key]
            R_0 = ttf.rotation_matrix(rotation_angle, rotation_axis)

            R_90 = np.matmul(R_0, ttf.rotation_matrix(math.pi / 2, axes[up_axis_key]))
            R_180 = np.matmul(R_90, ttf.rotation_matrix(math.pi / 2, axes[up_axis_key]))
            R_270 = np.matmul(R_180, ttf.rotation_matrix(math.pi / 2, axes[up_axis_key]))

            # store extents
            extents = list()
            for i in range(2):
                extents.append(
                    [axis_indices[rotation_axis_key], axis_indices[remaining_axis_key], axis_indices[up_axis_key]])
                extents.append(
                    [axis_indices[remaining_axis_key], axis_indices[rotation_axis_key], axis_indices[up_axis_key]])

            # store quaternions
            q_s = list()
            q_s.append(ttf.quaternion_from_matrix(R_0))
            q_s.append(ttf.quaternion_from_matrix(R_90))
            q_s.append(ttf.quaternion_from_matrix(R_180))
            q_s.append(ttf.quaternion_from_matrix(R_270))

            all_q_s[up_axis_key] = q_s
            all_extents[up_axis_key] = extents

        return all_q_s, all_extents

    def _get_class_specific_orientations_and_extents(self, cannonincal_qs, cannonical_extents):

        self._canonical_quaternions = dict()
        self._canonical_extents = dict()

        for class_id in self._class_ids:

            self._canonical_quaternions[class_id] = dict()
            self._canonical_extents[class_id] = dict()
            for up_axis_key in self._class_up_axis_keys:
                self._canonical_quaternions[class_id][up_axis_key] = cannonincal_qs[up_axis_key]
                self._canonical_extents[class_id][up_axis_key] = cannonical_extents[up_axis_key]

    def _get_object_poses(self, class_id, position_on_table):
        object_quaternions = list()
        object_extents = list()
        for quaternions in self._canonical_quaternions[class_id].values():
            object_quaternions += quaternions
        for extents in self._canonical_extents[class_id].values():
            object_extents += extents
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


if __name__ == '__main__':
    import objslampp.datasets.ycb_video as ycb_video_dataset
    object_models = ycb_video_dataset.YCBVideoModels()

    opi = ObjectPoseInterface(object_models)
    robot_to_object_mat = np.eye(4)
    robot_poses = opi.get_robot_poses(1, robot_to_object_mat, [0,0])

