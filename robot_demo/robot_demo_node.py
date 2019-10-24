import math
import numpy as np
import robot_demo.general_kinematics as gk
import rospy
import tf
from robot_demo.robot_interface import RobotInterface
from ros_objslampp_msgs.msg import ObjectPoseArray


class RobotDemo:

    def __init__(self):
        rospy.init_node('robot_demo')
        self._robot_interface = RobotInterface()
        self._all_objects_collected = False

        self._define_robot_poses()
        self._object_pose_found = False

        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf.TransformBroadcaster()

        self._obj_poses_sub = rospy.Subscriber('/camera/without_occupancy/singleview_3d_pose_estimation/output', ObjectPoseArray, self._object_poses_callback)

    # Object Pose Functions #
    # ----------------------#

    def _object_poses_callback(self, object_poses):

        # decompose pose array
        object_poses_header = object_poses.header
        object_pose = object_poses.poses[-1]
        pose = object_pose.pose

        position = pose.position
        orientation = pose.orientation

        # construct object frame transform
        translation = (position.x, position.y, position.z)
        rotation = (orientation.x, orientation.y, orientation.z, orientation.w)
        ros_time = object_poses_header.stamp
        child_frame = 'object_frame'
        parent_frame = 'camera_color_optical_frame'

        # publish transform
        self._tf_broadcaster.sendTransform(translation, rotation, ros_time, child_frame, parent_frame)

        self._object_pose_found = True

    def _get_object_pose(self):
        common_time = None
        while True:
            try:
                common_time = self._tf_listener.getLatestCommonTime('map', 'object_frame')
                break
            except:
                continue
        object_transform_tuple = self._tf_listener.lookupTransform('map', 'object_frame', common_time)
        object_position = object_transform_tuple[0]
        return np.concatenate((np.array(object_position), self._robot_interface.home_pose[3:]), -1)

    # Robot Functions #
    # ----------------#

    def _define_robot_poses(self):

        x_offset = 0

        z0 = 0.1

        x1 = 0.15
        y1 = 0.15
        z1 = 0
        angle1 = math.pi/8

        x2 = 0.1
        y2 = 0.1
        z2 = -0.05
        angle2 = math.pi/12

        z3 = -0.1

        # define robot poses
        robot_position_offsets = [np.array([x_offset, 0, z0]),

                                  np.array([x_offset, 0, z1]),
                                  np.array([x_offset+ x1, 0, z1]),
                                  np.array([x_offset + x1, y1, z1]),
                                  np.array([x_offset, y1, z1]),
                                  np.array([x_offset-x1, y1, z1]),
                                  np.array([x_offset-x1, 0, z1]),
                                  np.array([x_offset-x1, -y1, z1]),
                                  np.array([x_offset, -y1, z1]),
                                  np.array([x_offset + x1, -y1, z1]),

                                  np.array([x_offset, 0, z2]),
                                  np.array([x_offset + x2, 0, z2]),
                                  np.array([x_offset + x2, y2, z2]),
                                  np.array([x_offset, y2, z2]),
                                  np.array([x_offset-x2, y2, z2]),
                                  np.array([x_offset-x2, 0, z2]),
                                  np.array([x_offset-x2, -y2, z2]),
                                  np.array([x_offset, -y2, z2]),
                                  np.array([x_offset + x2, -y2, z2]),

                                  np.array([x_offset, 0, z3])]

        robot_rotation_vectors = [np.array([0, 0, 0]),

                                  np.array([0, 0, 0]),
                                  np.array([0, 1, 0])*angle1,
                                  np.array([-0.5**0.5, 0.5**0.5, 0])*angle1,
                                  np.array([-1, 0, 0])*angle1,
                                  np.array([-0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([0, -1, 0])*angle1,
                                  np.array([0.5**0.5, -0.5**0.5, 0])*angle1,
                                  np.array([1, 0, 0])*angle1,
                                  np.array([0.5**0.5, 0.5**0.5, 0])*angle1,

                                  np.array([0, 0, 0]),
                                  np.array([0, 1, 0]) * angle2,
                                  np.array([-0.5 ** 0.5, 0.5 ** 0.5, 0]) * angle2,
                                  np.array([-1, 0, 0]) * angle2,
                                  np.array([-0.5 ** 0.5, -0.5 ** 0.5, 0]) * angle2,
                                  np.array([0, -1, 0]) * angle2,
                                  np.array([0.5 ** 0.5, -0.5 ** 0.5, 0]) * angle2,
                                  np.array([1, 0, 0]) * angle2,
                                  np.array([0.5 ** 0.5, 0.5 ** 0.5, 0]) * angle2,

                                  np.array([0, 0, 0])]

        robot_quaternion_offsets = [gk.rotation_vector_to_quaternion(np, aa) for aa in robot_rotation_vectors]

        robot_positions = [self._robot_interface.home_pose[0:3] + offset for offset in robot_position_offsets]

        robot_quaternions = [gk.hamilton_product(np, self._robot_interface.home_pose[3:], qt) for qt in robot_quaternion_offsets]

        self._robot_poses = [np.concatenate((pos, quat),-1) for pos, quat in zip(robot_positions, robot_quaternions)]

    def _initialization_motion(self):
        self._robot_interface.move_to_home()
        for robot_pose in self._robot_poses:
            self._robot_interface.set_end_effector_quaternion_pose(robot_pose)

    def _move_robot_over_table(self):
        self._robot_interface.move_to_home()

    def _move_robot_to_pre_grasp_pose(self, object_pose):
        robot_pose = object_pose.copy()
        robot_pose[2] += 0.055
        self._robot_interface.set_end_effector_quaternion_pose(robot_pose)

    def _move_robot_until_force_feedback(self):
        pass

    def _suction_grip_object(self):
        pass

    def _place_object_on_table(self):
        pass

    # Object Checking #
    # ----------------#

    def _check_if_all_objects_collected(self):
        self._all_objects_collected = True if input('were all objects collected? [y/n]' ) == 'y' else False

    def run(self):
        self._initialization_motion()

        print('waiting for first object pose')
        while not self._object_pose_found:
            continue
        print('found first object pose')

        while not self._all_objects_collected:

            self._move_robot_over_table()
            object_pose = self._get_object_pose()
            self._move_robot_to_pre_grasp_pose(object_pose)
            self._move_robot_until_force_feedback()
            self._suction_grip_object()
            self._move_robot_to_pre_grasp_pose(object_pose)
            self._place_object_on_table()
            self._move_robot_over_table()
            self._check_if_all_objects_collected()


def main():
    robot_demo = RobotDemo()
    robot_demo.run()


if __name__ == '__main__':
    main()
