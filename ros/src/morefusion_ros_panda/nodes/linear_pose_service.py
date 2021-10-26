#!/usr/bin/env python

from actionlib_msgs.msg import GoalStatusArray
from morefusion_ros_panda.srv import MoveToPose
from morefusion_ros_panda.srv import MoveToPoseResponse
from moveit_commander import MoveGroupCommander
from moveit_commander import RobotCommander
import rospy

commander = MoveGroupCommander("panda_arm")
robot = RobotCommander()


def handle_move_to_pose(req):

    pose_goal = commander.get_current_pose()

    for i in range(len(req.goal_poses)):

        pose_goal.pose = req.goal_poses[i]
        if req.pure_translation:
            pose_goal.pose.orientation = (
                commander.get_current_pose().pose.orientation
            )
        if req.pure_rotation:
            pose_goal.pose.position = (
                commander.get_current_pose().pose.position
            )

        cartesian_plan, fraction = commander.compute_cartesian_path(
            [pose_goal.pose], 0.01, 0.0, req.avoid_collisions
        )
        scaled_cartesian_plan = commander.retime_trajectory(
            robot.get_current_state(), cartesian_plan, req.velocity_scaling
        )

    success = commander.execute(scaled_cartesian_plan, wait=True)

    return MoveToPoseResponse(success, commander.get_current_pose().pose)


def main():

    rospy.init_node("linear_move_to_pose_server", anonymous=True)
    rospy.wait_for_message("move_group/status", GoalStatusArray)
    rospy.Service("move_to_pose_linearly", MoveToPose, handle_move_to_pose)
    rospy.spin()


if __name__ == "__main__":
    main()
