import math
import robot_demo.general_kinematics as gk
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Quaternion, Pose
from ros_objslampp_srvs.srv import MoveToHome, MoveToPose


class RobotInterface:

    def __init__(self):
        self._define_home_pose()
        self._joint_angle_sub = rospy.Subscriber('/joint_angles', Float64MultiArray, self._joint_angle_callback,
                                                 queue_size=1)
        self._ef_pose_sub = rospy.Subscriber('/end_effector_pose', Float64MultiArray, self._ef_pose_callback,
                                                 queue_size=1)

        self._joint_angles = None
        self._ef_pose = None

    def _joint_angle_callback(self, data):
        self._joint_angles = data.data[1:]

    def _ef_pose_callback(self, data):
        self._ef_pose = data.data[1:]

    def _define_home_pose(self):
        self._home_position = np.array([0.3, 0, 0.6])
        q1 = gk.quaternion_from_vector_and_angle(np, [1, 0, 0], math.pi)
        q2 = gk.quaternion_from_vector_and_angle(np, [0, 0, 1], -math.pi/4)
        self._home_quaternion = gk.hamilton_product(np, q1, q2)
        self.home_pose = np.concatenate((self._home_position, self._home_quaternion), -1)

    def move_to_home(self, velocity_scaling=1.0, acceleration_scaling=1.0):
        rospy.wait_for_service('move_to_home')
        move_to_home = rospy.ServiceProxy('move_to_home', MoveToHome)
        response = move_to_home(velocity_scaling, acceleration_scaling)
        return response.home_reached

    # end effector pose #

    # getters

    def get_end_effector_position(self):
        pass

    def get_end_effector_quaternion(self):
        pass

    def get_end_effector_quaternion_pose(self):
        return self._ef_pose

    # setters

    def set_end_effector_position(self, position, velocity_scaling=1.0, acceleration_scaling=1.0):
        rospy.wait_for_service('move_to_position_linearly')
        move_to_position = rospy.ServiceProxy('move_to_position_linearly', MoveToPose)
        quaternion = Quaternion(0,0,0,1)
        pose = Pose()
        pose.position = position
        pose.orientation = quaternion
        response = move_to_position(pose, velocity_scaling, acceleration_scaling)
        return response.pose_reached

    def set_end_effector_quaternion(self, quaternion):
        pass

    def set_end_effector_quaternion_pose(self, pose, velocity_scaling=1.0, acceleration_scaling=1.0):
        rospy.wait_for_service('move_to_pose_linearly')
        move_to_pose = rospy.ServiceProxy('move_to_pose_linearly', MoveToPose)
        response = move_to_pose(pose, velocity_scaling, acceleration_scaling)
        return response.pose_reached

    def set_end_effector_quaternion_pointing_pose(self, pose, velocity_scaling=1.0, acceleration_scaling=1.0):
        rospy.wait_for_service('/pointing_pose_service/move_to_pointing_pose')
        move_to_pose = rospy.ServiceProxy('/pointing_pose_service/move_to_pointing_pose', MoveToPose)
        response = move_to_pose(pose, velocity_scaling, acceleration_scaling)
        return response.pose_reached

    # joint angles #

    # getters

    def get_joint_angles(self):
        return self._joint_angles

    # setters

    def set_joint_angles(self, joint_angles):
        pass