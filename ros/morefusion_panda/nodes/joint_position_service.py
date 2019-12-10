#!/usr/bin/env python

import rospy
from morefusion_panda.srv import MoveToJointPosition
from morefusion_panda.srv import MoveToJointPositionResponse
from moveit_commander import MoveGroupCommander, RobotCommander
from actionlib_msgs.msg import GoalStatusArray

commander = MoveGroupCommander('panda_arm')
robot = RobotCommander()


def handle_move_to_joint_position(req):
    joint_goal = req.joint_position
    commander.set_max_velocity_scaling_factor(req.velocity_scaling)
    commander.set_max_acceleration_scaling_factor(req.acceleration_scaling)
    success = commander.go(joint_goal, wait=True)
    return MoveToJointPositionResponse(
        success, commander.get_current_joint_values()
    )


def main():

    rospy.init_node('move_to_joint_position_server', anonymous=True)
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    rospy.Service(
        'move_to_joint_position',
        MoveToJointPosition,
        handle_move_to_joint_position,
    )
    rospy.spin()


if __name__ == '__main__':
    main()
