import numpy as np


def from_ros_transform(transform):
    quaternion = from_ros_orientation(transform.rotation)
    translation = from_ros_vector3(transform.translation)
    return quaternion, translation


def from_ros_pose(pose):
    quaternion = from_ros_orientation(pose.orientation)
    translation = from_ros_vector3(pose.position)
    return quaternion, translation


def from_ros_vector3(vector3):
    vector3 = np.array([
        vector3.x,
        vector3.y,
        vector3.z,
    ], dtype=np.float32)
    return vector3


def from_ros_orientation(orientation):
    quaternion = np.array([
        orientation.w,
        orientation.x,
        orientation.y,
        orientation.z,
    ], dtype=np.float32)
    return quaternion
