import rospy
import math as _math

from geometry_msgs.msg import Pose as _Pose
from geometry_msgs.msg import Vector3 as _Vector3
from geometry_msgs.msg import Quaternion as _Quaternion

from ros_objslampp_srvs.srv import MoveToPose as _MoveToPose
from ros_objslampp_srvs.srv import SetSuction as _SetSuction
from ros_objslampp_srvs.srv import MoveToJointPosition as _MoveToJointPosition


# move to home #


def move_to_home(velocity_scaling=0.5, acceleration_scaling=0.5, must_succeed=True):

    rospy.wait_for_service('/move_to_joint_position')
    move_to_joint_position = rospy.ServiceProxy('/move_to_joint_position', _MoveToJointPosition)

    home_joint_position = [0.0049455467613561555, 0.20376276994262754, 0.008827571982877296, -1.021473878492389,
                           0.02799238988916705, 1.246361540502972, 0.7944748621281127]

    try:
        response = move_to_joint_position(home_joint_position, velocity_scaling, acceleration_scaling)
    except:
        if must_succeed:
            raise Exception('Could not move robot to move.')
        return False, []
    if must_succeed and not response.success:
        raise Exception('Could not move robot to move.')
    return response.success, response.joint_position_reached


# end effector pose #


def set_end_effector_position_linearly(position, velocity_scaling=0.5, acceleration_scaling=0.5, must_succeed=True):

    rospy.wait_for_service('/move_to_pose_linearly')
    move_to_position = rospy.ServiceProxy('/move_to_pose_linearly', _MoveToPose)

    quaternion = _Quaternion(0,0,0,1)
    pose = _Pose()
    pose.position = position
    pose.orientation = quaternion

    try:
        response = move_to_position([pose], [], [], '', velocity_scaling, acceleration_scaling, True, False, False)
    except:
        if must_succeed:
            raise Exception('Could set robot end effector position linearly.')
        return False, _Pose()
    if must_succeed and not response.success:
        raise Exception('Could set robot end effector position linearly.')
    return response.success, response.pose_reached


def set_end_effector_quaternion_pose_linearly(pose, velocity_scaling=0.5, acceleration_scaling=0.5, must_succeed=True):

    rospy.wait_for_service('/move_to_pose_linearly')
    move_to_position = rospy.ServiceProxy('/move_to_pose_linearly', _MoveToPose)

    try:
        response = move_to_position([pose], [], [], '', velocity_scaling, acceleration_scaling, False, False, False)
    except:
        if must_succeed:
            raise Exception('Could set robot end effector quaternion pose linearly.')
        return False, _Pose()
    if must_succeed and not response.success:
        raise Exception('Could set robot end effector quaternion pose linearly.')
    return response.success, response.pose_reached


def set_end_effector_quaternion_pose(poses, velocity_scaling=0.5, acceleration_scaling=0.5, must_succeed=True):

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

    try:
        response = move_to_pose(poses, [position_contraint]*len(poses), [orientation_contraint]*len(poses),
                                'panda_suction_cup', velocity_scaling, acceleration_scaling, False, False, True)
    except:
        if must_succeed:
            raise Exception('Could set robot end effector quaternion pose.')
        return False, _Pose()
    if must_succeed and not response.success:
        raise Exception('Could set robot end effector quaternion pose.')
    return response.success, response.pose_reached


def set_end_effector_quaternion_pointing_pose(pose, velocity_scaling=0.5, acceleration_scaling=0.5, must_succeed=True):

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

    try:
        response = move_to_pose([pose], [position_contraint], [orientation_contraint], 'panda_suction_cup',
                                velocity_scaling, acceleration_scaling, False, False, True)
    except:
        if must_succeed:
            raise Exception('Could set robot end effector quaternion pointing pose.')
        return False, _Pose()
    if must_succeed and not response.success:
        raise Exception('Could set robot end effector quaternion pointing pose.')
    return response.success, response.pose_reached


# suction gripper #


def set_suction_state(state, must_succeed=True):
    rospy.wait_for_service('/set_suction')
    set_suction = rospy.ServiceProxy('/set_suction', _SetSuction)
    try:
        response = set_suction(state)
    except:
        if must_succeed:
            raise Exception('Could set robot suction state.')
        return False
    if must_succeed and not response.success:
        raise Exception('Could set robot suction state.')
    return response.success
