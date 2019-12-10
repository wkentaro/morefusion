#!/bin/bash

set -e

ROSOBJSLAMPP_PREFIX=$HOME/ros_objslampp

unset PYTHONPATH
unset CMAKE_PREFIX_PATH

source /opt/ros/kinetic/setup.bash

set -x

mkdir -p $ROSOBJSLAMPP_PREFIX/src
cd $ROSOBJSLAMPP_PREFIX
catkin init

if [ ! -e $ROSOBJSLAMPP_PREFIX/src/.rosinstall ]; then
  ln -s $ROSOBJSLAMPP_PREFIX/src/objslampp/ros/rosinstall $ROSOBJSLAMPP_PREFIX/src/.rosinstall
  (cd $ROSOBJSLAMPP_PREFIX/src && wstool up)
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

catkin build ros_objslampp_panda

set +x
source /opt/ros/kinetic/setup.bash
source $ROSOBJSLAMPP_PREFIX/devel/setup.bash
set -x
