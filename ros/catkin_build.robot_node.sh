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
catkin config --whitelist \
  franka_control \
  franka_description \
  franka_example_controllers \
  franka_gripper \
  franka_hw \
  franka_msgs \
  franka_ros \
  franka_visualization \
  franka_description_custom \
  panda_moveit_config_custom \
  ros_objslampp_panda

catkin build

set +x
source /opt/ros/kinetic/setup.bash
source $ROSOBJSLAMPP_PREFIX/devel/setup.bash
set -x
