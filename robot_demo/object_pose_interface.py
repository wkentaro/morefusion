import math
import numpy as np
from robot_demo import general_kinematics as gk
import trimesh
import trimesh.transformations as ttf
from geometry_msgs.msg import Pose


class ObjectPoseInterface:

    def __init__(self, object_models):
        self._floor_z = 0.11
        self._object_models = object_models
        self._class_names = self._object_models.class_names
        self._class_ids = range(1,len(self._object_models.class_names))

        # define z rotation angles
        self._z_rotations = dict()
        self._z_rotations[1] = -13*math.pi/180  # master_chef_can
        self._z_rotations[2] = 0                # cracker box
        self._z_rotations[3] = 0                # sugar box
        self._z_rotations[4] = -20*math.pi/180  # soup can
        self._z_rotations[5] = 21*math.pi/180   # mustard bottle
        self._z_rotations[6] = 0                # tuna can
        self._z_rotations[7] = -29*math.pi/180  # pudding box
        self._z_rotations[8] = -13*math.pi/180  # gelatin box
        self._z_rotations[9] = -2*math.pi/180   # potted meat can
        self._z_rotations[10] = 0               # banana
        self._z_rotations[11] = -43*math.pi/180 # pitcher base
        self._z_rotations[12] = 0               # bleach cleanser
        self._z_rotations[13] = 0               # bowl
        self._z_rotations[14] = 4*math.pi/180   # mug
        self._z_rotations[15] = 0               # power drill
        self._z_rotations[16] = 13*math.pi/180  # wood block
        self._z_rotations[17] = 17*math.pi/180  # scissors
        self._z_rotations[18] = 0               # large marker
        self._z_rotations[19] = -8*math.pi/180  # large clamp
        self._z_rotations[20] = -7*math.pi/180  # extra large clamp
        self._z_rotations[21] = 0               # foam brick

        # define class specific up axes
        self._class_up_axis_keys = dict()
        self._class_up_axis_keys[1] = ['z+', 'z-'] # master_chef_can
        self._class_up_axis_keys[2] = ['x+', 'x-'] # cracker box
        self._class_up_axis_keys[3] = ['x+', 'x-'] # sugar box
        self._class_up_axis_keys[4] = ['z+', 'z-'] # soup can
        self._class_up_axis_keys[5] = ['y+', 'y-', 'z+'] # mustard bottle
        self._class_up_axis_keys[6] = ['z+', 'z-'] # tuna can
        self._class_up_axis_keys[7] = ['z+', 'z-'] # pudding box
        self._class_up_axis_keys[8] = ['z+', 'z-'] # gelatin box
        self._class_up_axis_keys[9] = ['y+', 'y-', 'z+', 'z-'] # potted meat can
        self._class_up_axis_keys[10] = ['z+', 'z-'] # banana
        self._class_up_axis_keys[11] = ['z+'] # pitcher base
        self._class_up_axis_keys[12] = ['y+', 'y-', 'z+'] # bleach cleanser
        self._class_up_axis_keys[13] = ['z+', 'z-'] # bowl
        self._class_up_axis_keys[14] = ['z+', 'z-'] # mug
        self._class_up_axis_keys[15] = ['z+', 'z-'] # power drill
        self._class_up_axis_keys[16] = ['x+', 'x-', 'y+', 'y-'] # wood block
        self._class_up_axis_keys[17] = ['z+', 'z-'] # scissors
        self._class_up_axis_keys[18] = ['x+', 'x-', 'z+', 'z-'] # large marker
        self._class_up_axis_keys[19] = ['z+', 'z-'] # large clamp
        self._class_up_axis_keys[20] = ['z+', 'z-'] # extra large clamp
        self._class_up_axis_keys[21] = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-'] # foam brick

        # Define canonical models
        canonical_quaternions, canonical_extents = self._get_cannonical_object_rotation_mats_and_extents()
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

    def _get_cannonical_object_rotation_mats_and_extents(self):

        # final data container
        all_R_s = dict()
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
            # ToDo: change so that unaligned objects do not have over conservative extents, now the object is corrected
            extents = list()
            for i in range(2):
                extents.append(
                    [axis_indices[rotation_axis_key], axis_indices[remaining_axis_key], axis_indices[up_axis_key]])
                extents.append(
                    [axis_indices[remaining_axis_key], axis_indices[rotation_axis_key], axis_indices[up_axis_key]])

            # store quaternions
            R_s = list()
            for R in [R_0, R_90, R_180, R_270]:
                R_s.append(R)

            all_R_s[up_axis_key] = R_s
            all_extents[up_axis_key] = extents

        return all_R_s, all_extents

    def _get_class_specific_orientations_and_extents(self, cannonincal_Rs, cannonical_extents):

        self._canonical_quaternions = dict()
        self._canonical_extents = dict()

        for class_id in self._class_ids:

            self._canonical_quaternions[class_id] = dict()
            self._canonical_extents[class_id] = dict()

            R_z_fix = ttf.rotation_matrix(self._z_rotations[class_id], [0., 0., 1.])

            for up_axis_key in self._class_up_axis_keys[class_id]:

                q_s = list()

                for R in cannonincal_Rs[up_axis_key]:
                    R_tot = np.matmul(R, R_z_fix)
                    q = ttf.quaternion_from_matrix(R_tot)
                    q_s.append(np.concatenate((q[1:], q[0:1])))

                self._canonical_quaternions[class_id][up_axis_key] = q_s
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

