import math as _math
import robot_demo.general_kinematics as _gk
import numpy as _np
import rospy
from geometry_msgs.msg import Quaternion as _Quaternion
from geometry_msgs.msg import Pose as _Pose
from geometry_msgs.msg import Vector3 as _Vector3
from ros_objslampp_srvs.srv import MoveToPose as _MoveToPose
from ros_objslampp_srvs.srv import SetSuction as _SetSuction


# move to home #


_home_position = _np.array([0.6, 0, 0.6])
_q1 = _gk.quaternion_from_vector_and_angle(_np, [1, 0, 0], _math.pi)
_q2 = _gk.quaternion_from_vector_and_angle(_np, [0, 0, 1], -_math.pi/4)
_home_quaternion = _gk.hamilton_product(_np, _q1, _q2)
home_pose = _np.concatenate((_home_position, _home_quaternion), -1)


def move_to_home(velocity_scaling=0.75, acceleration_scaling=0.75):

    home_pose_msg = _Pose()

    home_pose_msg.position.x = home_pose[0]
    home_pose_msg.position.y = home_pose[1]
    home_pose_msg.position.z = home_pose[2]

    home_pose_msg.orientation.x = home_pose[3]
    home_pose_msg.orientation.y = home_pose[4]
    home_pose_msg.orientation.z = home_pose[5]
    home_pose_msg.orientation.w = home_pose[6]

    return set_end_effector_quaternion_pose_linearly(home_pose_msg, velocity_scaling, acceleration_scaling)


# end effector pose #


def set_end_effector_position_linearly(position, velocity_scaling=0.75, acceleration_scaling=0.75):

    rospy.wait_for_service('/move_to_pose_linearly')
    move_to_position = rospy.ServiceProxy('/move_to_pose_linearly', _MoveToPose)

    quaternion = _Quaternion(0,0,0,1)
    pose = _Pose()
    pose.position = position
    pose.orientation = quaternion

    response = move_to_position([pose], [], [], '', velocity_scaling, acceleration_scaling, True, False, False)

    if response.success:
        return response.pose_reached
    else:
        raise Exception('Unable to set robot position.')


def set_end_effector_quaternion_pose_linearly(pose, velocity_scaling=0.75, acceleration_scaling=0.75):

    rospy.wait_for_service('/move_to_pose_linearly')
    move_to_position = rospy.ServiceProxy('/move_to_pose_linearly', _MoveToPose)

    response = move_to_position([pose], [], [], '', velocity_scaling, acceleration_scaling, False, False, False)

    if response.success:
        return response.pose_reached
    else:
        raise Exception('Unable to set robot position.')


def set_end_effector_quaternion_pose(poses, velocity_scaling=0.75, acceleration_scaling=0.75):

    rospy.wait_for_service('/pose_service/move_to_pose')
    move_to_pose = rospy.ServiceProxy('/pose_service/move_to_pose', _MoveToPose)

    position_contraint = _Vector3()
    position_contraint.x = 0.001
    position_contraint.y = 0.001
    position_contraint.z = 0.001

    orientation_contraint = _Vector3()
    orientation_contraint.x = 0.001
    orientation_contraint.y = 0.001
    orientation_contraint.z = 0.001

    response = move_to_pose(poses, [position_contraint]*len(poses), [orientation_contraint]*len(poses),
                            'panda_suction_cup', velocity_scaling, acceleration_scaling, False, False, True)
    if response.success:
        return response.pose_reached
    else:
        raise Exception('Unable to set robot pose.')


def set_end_effector_quaternion_pointing_pose(pose, velocity_scaling=0.75, acceleration_scaling=0.75):

    rospy.wait_for_service('/pose_service/move_to_pose')
    move_to_pose = rospy.ServiceProxy('/pose_service/move_to_pose', _MoveToPose)

    position_contraint = _Vector3()
    position_contraint.x = 0.001
    position_contraint.y = 0.001
    position_contraint.z = 0.001

    orientation_contraint = _Vector3()
    orientation_contraint.x = 0.001
    orientation_contraint.y = 0.001
    orientation_contraint.z = 2*_math.pi

    response = move_to_pose([pose], [position_contraint], [orientation_contraint], 'panda_suction_cup',
                            velocity_scaling, acceleration_scaling, False, False, True)
    if response.success:
        return response.pose_reached
    else:
        raise Exception('Unable to set robot pointing pose.')


# suction gripper #


def set_suction_state(state):
    rospy.wait_for_service('/set_suction')
    set_suction = rospy.ServiceProxy('/set_suction', _SetSuction)
    response = set_suction(state)
    if response.suction_set:
        return True
    else:
        raise Exception('Unable to set robot suction.')
