#!/usr/bin/env python

import rospy
from franka_moveit_custom.srv import MoveToHome, MoveToHomeResponse
from moveit_commander import MoveGroupCommander
from actionlib_msgs.msg import GoalStatusArray

commander = MoveGroupCommander('panda_arm')


def handle_move_to_home(req):
    commander.set_max_velocity_scaling_factor(req.velocity_scaling)
    commander.set_max_acceleration_scaling_factor(req.acceleration_scaling)
    commander.set_named_target('ready')
    success = commander.go(wait=True)
    return MoveToHomeResponse(success)


def main():

    rospy.init_node('move_to_home_server', anonymous=True)
    rospy.wait_for_message('move_group/status', GoalStatusArray)
    rospy.Service('move_to_home', MoveToHome, handle_move_to_home)
    rospy.spin()


if __name__ == '__main__':
    main()
