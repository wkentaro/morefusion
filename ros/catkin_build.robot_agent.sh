#!/bin/bash

set -e

PY_ROOT_PREFIX=$HOME/ros_morefusion/src/morefusion
ROS_ROOT_PREFIX=$HOME/ros_morefusion

if [ ! -d $PY_ROOT_PREFIX ]; then
  echo "Please install morefusion to $PY_ROOT_PREFIX"
  exit 1
fi

if [ ! -e $PY_ROOT_PREFIX/.anaconda3/bin/activate ]; then
  echo "Please run 'make install' in morefusion"
  exit 1
fi

unset PYTHONPATH
unset CMAKE_PREFIX_PATH

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

if [ ! -e $ROS_ROOT_PREFIX/src/.rosinstall ]; then
  ln -s $ROS_ROOT_PREFIX/src/morefusion/ros/rosinstall $ROS_ROOT_PREFIX/src/.rosinstall
  (cd $ROS_ROOT_PREFIX/src && wstool up)
fi

if [ ! -e $ROS_ROOT_PREFIX/.autoenv.zsh ]; then
  cp $PY_ROOT_PREFIX/ros/template.autoenv.zsh $ROS_ROOT_PREFIX/.autoenv.zsh
fi
if [ ! -e $ROS_ROOT_PREFIX/.autoenv_leave.zsh ]; then
  cp $PY_ROOT_PREFIX/ros/template.autoenv_leave.zsh $ROS_ROOT_PREFIX/.autoenv_leave.zsh
fi

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
