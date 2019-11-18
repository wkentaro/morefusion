import time

import numpy as np

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

        self._wait_for_service('/move_to_joint_position')
        self._wait_for_service('/set_suction')

    def _wait_for_service(self, name):
        rospy.loginfo(f"Waiting for the service '{name}'")
        rospy.wait_for_service(name)
        rospy.loginfo(f"Found the service '{name}'")

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'scale_vel={self.scale_vel}, '
                f'scale_accel={self.scale_accel})')

    def _set_joint_position(self, position, scale_vel=None, scale_accel=None):
        if scale_vel is None:
            scale_vel = self.scale_vel
        if scale_accel is None:
            scale_accel = self.scale_accel

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

    def run_scanning_motion(self):
        joint_positions = [
            (0.08224239694750395, -0.4015997108409278, -0.49498474810834514, -2.3139969589166474, 0.33205986473317717, 2.125227750738227, 0.7763856610548164),   # rb
            (-0.10939621453640754, 0.49313024987672505, -0.5767242879783897, -1.2114025848957364, 0.6707009797625332, 1.3963332301586997, 0.15708699394183026),  # rm
            (0.1698192655535584, -0.0186732256553675, -0.2541714021814522, -1.84699400115707, 0.021888164495122868, 1.9280040739047786, -0.6447974156447582),    # mm
            (0.17259695036474026, -0.847359646914298, -0.3413624777681859, -2.697739678432234, -0.2500084476338492, 2.327152996407615, -0.7101045701824773),     # mb
            (-0.3075652822371115, -1.1519815396593327, 0.991405422754455, -2.6923269817676725, 0.29809131103778574, 2.180580173133383, 0.292405972569353),       # lb
            (-1.4505077696237818, -1.0788255915538485, 1.3209980016478426, -1.5838640493445924, 0.6088219937132866, 1.562294041660097, 0.7018719817101955),      # lm
            (-1.9088786778209579, -0.5086115553877906, 1.6564622810977294, -1.5827366967465282, 0.5234668570094723, 1.61768960422073, 0.5276833137762216),  # mm2
        ]
        for i in range(len(joint_positions)):
            if i == 0:
                jp = joint_positions[i]
                self._set_joint_position(jp, 1, 1)
                time.sleep(0.5)
            else:
                jp_prev = joint_positions[i - 1]
                jp_next = joint_positions[i]
                for jp in np.linspace(jp_prev, jp_next, 4):  # 2 middle points
                    self._set_joint_position(jp, 1, 1)
                    time.sleep(0.5)


if __name__ == '__main__':
    rospy.init_node('robot_interface')
    ri = RobotInterface()
