import rospy

from ros_objslampp_srvs.srv import MoveToJointPosition
from ros_objslampp_srvs.srv import SetSuction


class RobotInterface:

    _joint_positions = {
        'reset': (
            0.0350888149879746,
            -0.9124876247539854,
            -0.09572808180781056,
            -2.040543374092091,
            -0.1354376670387056,
            1.1432695249186622,
            0.7123907900742359,
        ),
        'overlook': (
            0.0049455467613561555,
            0.20376276994262754,
            0.008827571982877296,
            -1.021473878492389,
            0.02799238988916705,
            1.246361540502972,
            0.7944748621281127,
        ),
    }

    def __init__(self):
        self.scale_vel = 0.5
        self.scale_accel = 0.5

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'scale_vel={self.scale_vel}, '
                f'scale_accel={self.scale_accel})')

    def _wait_for_service(self, name):
        rospy.loginfo(f"Waiting for the service '{name}'")
        rospy.wait_for_service(name)
        rospy.loginfo(f"Found the service '{name}'")

    def _set_joint_position(self, position, scale_vel=None, scale_accel=None):
        if scale_vel is None:
            scale_vel = self.scale_vel
        if scale_accel is None:
            scale_accel = self.scale_accel

        self._wait_for_service('/move_to_joint_position')
        client = rospy.ServiceProxy(
            '/move_to_joint_position', MoveToJointPosition
        )
        response = client(position, scale_vel, scale_accel)

        if not response.success:
            raise RuntimeError(
                f'Failed to move to the joint position: {position}'
            )
        return response.joint_position_reached

    def _set_suction_on(self, on):
        self._wait_for_service('/set_suction')
        client = rospy.ServiceProxy('/set_suction', SetSuction)
        response = client(on)

        if not response.success:
            raise RuntimeError(
                f'Failed to set suction on: {on}'
            )

    # -------------------------------------------------------------------------

    def set_end_effector_position_linearly(self, *args, **kwargs):
        from robot_interface import set_end_effector_position_linearly
        return set_end_effector_position_linearly(*args, **kwargs)

    def set_end_effector_quaternion_pose_linearly(self, *args, **kwargs):
        from robot_interface import set_end_effector_quaternion_pose_linearly
        return set_end_effector_quaternion_pose_linearly(*args, **kwargs)

    def set_end_effector_quaternion_pose(self, *args, **kwargs):
        from robot_interface import set_end_effector_quaternion_pose
        return set_end_effector_quaternion_pose(*args, **kwargs)

    def set_end_effector_quaternion_pointing_pose(self, *args, **kwargs):
        from robot_interface import set_end_effector_quaternion_pointing_pose
        return set_end_effector_quaternion_pointing_pose(*args, **kwargs)

    def grasp(self):
        return self._set_suction_on(on=True)

    def ungrasp(self):
        return self._set_suction_on(on=False)

    def move_to_overlook_pose(self, scale_vel=None, scale_accel=None):
        return self._set_joint_position(
            self._joint_positions['overlook'],
            scale_vel,
            scale_accel,
        )

    def move_to_reset_pose(self, scale_vel=None, scale_accel=None):
        return self._set_joint_position(
            self._joint_positions['reset'],
            scale_vel,
            scale_accel,
        )


if __name__ == '__main__':
    rospy.init_node('robot_interface')
    ri = RobotInterface()
