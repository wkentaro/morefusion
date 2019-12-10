#!/bin/bash

ROSOBJSLAMPP_PREFIX=$HOME/ros_objslampp

cd $ROSOBJSLAMPP_PREFIX

set -e
set -x

$ROSOBJSLAMPP_PREFIX/src/objslampp/ros/install_realsense.sh

rosdep init
rosdep update

rosdep install --from-path src/objslampp/ros --skip-keys 'realsense2_camera orb_slam2_ros franka_description_custom'
rosdep install --from-path src/jsk-ros-pkg/jsk_recognition/jsk_recognition_utils
