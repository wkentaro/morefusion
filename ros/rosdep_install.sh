#!/bin/bash

ROS_ROOT_PREFIX=$HOME/ros_morefusion

cd $ROS_ROOT_PREFIX

set -e
set -x

$ROS_ROOT_PREFIX/src/morefusion/ros/install_realsense.sh

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update

rosdep install --from-path src/morefusion/ros --skip-keys 'realsense2_camera orb_slam2_ros morefusion_panda'
rosdep install --from-path src/jsk-ros-pkg/jsk_recognition/jsk_recognition_utils --skip-keys 'python-chainer-pip'
