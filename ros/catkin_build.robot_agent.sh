#!/bin/bash

set -e

HERE="$(dirname $(realpath ${BASH_SOURCE[0]}))"
source $HERE/.init.sh

source $PY_ROOT_PREFIX/.anaconda3/bin/activate
source /opt/ros/kinetic/setup.bash

set -x

pip install catkin_pkg
pip install rospkg
pip install empy
pip install netifaces

mkdir -p $ROS_ROOT_PREFIX/src
cd $ROS_ROOT_PREFIX
catkin init

catkin config --merge-devel \
              -DPYTHON_EXECUTABLE=$PY_ROOT_PREFIX/.anaconda3/bin/python \
              -DPYTHON_INCLUDE_DIR=$PY_ROOT_PREFIX/.anaconda3/include/python3.7m \
              -DPYTHON_LIBRARY=$PY_ROOT_PREFIX/.anaconda3/lib/libpython3.7m.so \
              --cmake-args -DCMAKE_BUILD_TYPE=Release -DOCTOMAP_OMP=1
catkin config --blacklist \
  checkerboard_detector \
  jsk_network_tools \
  jsk_tools \
  jsk_recognition_msgs \
  imagesift \
  image_view2 \
  jsk_perception \
  jsk_pcl_ros \
  jsk_pcl_ros_utils \
  rosbag \
  rosbag_storage \
  franka_control \
  franka_gripper \
  franka_hw \
  franka_visualization
mkdir -p $ROS_ROOT_PREFIX/devel/lib/python3/dist-packages
ln -fs $PY_ROOT_PREFIX/.anaconda3/lib/python3.7/site-packages/cv2 $ROS_ROOT_PREFIX/devel/lib/python3/dist-packages

catkin build cv_bridge

set +x
source $PY_ROOT_PREFIX/.anaconda3/bin/activate
source /opt/ros/kinetic/setup.bash
source $ROS_ROOT_PREFIX/devel/setup.bash
set -x

python -c 'import cv2'
python -c 'from cv_bridge.boost.cv_bridge_boost import getCvType'

catkin build morefusion_panda_ycb_video
