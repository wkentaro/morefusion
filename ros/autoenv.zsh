#!/usr/bin/env zsh

setup() {
  export ROS_PYTHON_VERSION=3

  source /opt/ros/kinetic/setup.zsh
  if [ -e $HERE/devel/setup.zsh ]; then
    source $HERE/devel/setup.zsh
  fi

  rosdefault
  # XXX: robot-agent needs to be hidden by robot-node to
  # reduce topic connections to avoid reflex error in Panda
  # rossetip

  CONDA_PREFIX=$HERE/src/morefusion/.anaconda3
  source $CONDA_PREFIX/bin/activate

  export WSTOOL_DEFAULT_WORKSPACE=$HERE/src

  show-ros
  echo "PYTHONPATH: $PYTHONPATH"
  echo "PYTHON: $(command which python)"
}

HERE=${0:a:h}
setup
