#!/bin/bash

set -e
set -x

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

ln -fs $PY_ROOT_PREFIX/ros/*.sh $ROS_ROOT_PREFIX/
ln -fs $PY_ROOT_PREFIX/ros/.autoenv*.zsh $ROS_ROOT_PREFIX/

if [ ! -e $ROS_ROOT_PREFIX/src/.rosinstall ]; then
  ln -s $ROS_ROOT_PREFIX/src/morefusion/ros/rosinstall $ROS_ROOT_PREFIX/src/.rosinstall
  (cd $ROS_ROOT_PREFIX/src && wstool up)
fi

set +x
set +e
