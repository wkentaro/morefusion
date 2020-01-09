#!/bin/bash

set -e

HERE="$(dirname $(realpath ${BASH_SOURCE[0]}))"
source $HERE/.init.sh

source /opt/ros/kinetic/setup.bash

set -x

mkdir -p $ROS_ROOT_PREFIX/src
cd $ROS_ROOT_PREFIX
catkin init

if [ ! -e $ROS_ROOT_PREFIX/src/.rosinstall ]; then
  ln -s $ROS_ROOT_PREFIX/src/morefusion/ros/rosinstall $ROS_ROOT_PREFIX/src/.rosinstall
  (cd $ROS_ROOT_PREFIX/src && wstool up)
fi

catkin config --merge-devel \
              --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --blacklist \
  checkerboard_detector \
  imagesift \
  image_view2 \
  jsk_network_tools \
  jsk_pcl_ros \
  jsk_pcl_ros_utils \
  jsk_perception \
  jsk_recognition_msgs \
  jsk_tools \
  message_filters \
  rosbag \
  rosbag_storage \
  rosconsole \
  roscpp \
  rosgraph \
  roslaunch \
  roslz4 \
  rosmaster \
  rosmsg \
  rosmsg \
  rosnode \
  rosout \
  rosparam \
  rospy \
  rosservice \
  rostest \
  rostopic \
  roswtf \
  tf \
  tf2 \
  tf2_eigen \
  tf2_geometry_msgs \
  tf2_msgs \
  tf2_py \
  tf2_ros \
  topic_tools \
  xmlrpcpp

catkin build morefusion_panda
