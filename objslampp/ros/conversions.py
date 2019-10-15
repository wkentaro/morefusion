import numpy as np


def pose_from_pose(pose_msg):
    quaternion = quaternion_from_orientation(pose_msg.orientation)
    translation = translation_from_position(pose_msg.position)
    return quaternion, translation


def translation_from_position(position):
    translation = np.array([
        position.x,
        position.y,
        position.z,
    ], dtype=np.float32)
    return translation


def quaternion_from_orientation(orientation):
    quaternion = np.array([
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
    ], dtype=np.float32)
    return quaternion
