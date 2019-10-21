#!/usr/bin/env zsh

rossetpanda() {
  rossetdefault 172.16.0.1
}

setup() {
  source /opt/ros/kinetic/setup.zsh
  if [ -e $HERE/devel/setup.zsh ]; then
    source $HERE/devel/setup.zsh
  fi

  CONDA_PREFIX=$HERE/src/objslampp/.anaconda3
  source $CONDA_PREFIX/bin/activate

  rosdefault
  rossetip

  show-ros
  echo "PYTHONPATH: $PYTHONPATH"
  echo "PYTHON: $(command which python)"
}

HERE=${0:a:h}
setup
