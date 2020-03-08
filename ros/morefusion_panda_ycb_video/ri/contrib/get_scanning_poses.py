import math

import numpy as np

from . import general_kinematics as gk


def get_scanning_poses():
    _home_position = np.array([0.5, 0, 0.6])
    _home_quaternion = gk.quaternion_from_vector_and_angle(
        np, [1, 0, 0], math.pi
    )
    home_pose = np.concatenate((_home_position, _home_quaternion), -1)

    x_offset = 0.15
    y_offset = 0.02

    x1 = 0.05
    y1 = 0.08
    z1 = -0.12

    # define robot poses
    robot_position_offsets = [
        # centre
        np.array([x_offset, y_offset, z1]),
        # top
        np.array([x_offset + x1, y_offset, z1]),
        # top right
        np.array([x_offset + x1, y_offset - y1, z1]),
        # right
        np.array([x_offset, y_offset - y1, z1]),
        # bottom right
        np.array([x_offset - x1, y_offset - y1, z1]),
        # bottom
        np.array([x_offset - x1, y_offset, z1]),
        # centre
        np.array([x_offset, y_offset, z1]),
        # top
        np.array([x_offset + x1, y_offset, z1]),
        # top left
        np.array([x_offset + x1, y_offset + y1, z1]),
        # left
        np.array([x_offset, y_offset + y1, z1]),
        # bottom left
        np.array([x_offset - x1, y_offset + y1, z1]),
        # centre
        np.array([x_offset, y_offset, z1]),
        np.array([x_offset, y_offset, 0]),
    ]

    robot_rotation_vectors = [np.array([0, 0, 0])] * len(
        robot_position_offsets
    )

    robot_quaternion_offsets = [
        gk.rotation_vector_to_quaternion(np, aa)
        for aa in robot_rotation_vectors
    ]
    robot_positions = [
        home_pose[0:3] + offset for offset in robot_position_offsets
    ]
    robot_quaternions = [
        gk.hamilton_product(np, home_pose[3:], qt)
        for qt in robot_quaternion_offsets
    ]
    robot_poses = [
        np.concatenate((pos, quat), -1)
        for pos, quat in zip(robot_positions, robot_quaternions)
    ]

    return robot_poses
