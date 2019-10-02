#!/usr/bin/env zsh

setup() {
  source /opt/ros/kinetic/setup.zsh
  if [ -e $HERE/devel/setup.zsh ]; then
    source $HERE/devel/setup.zsh
  fi

  CONDA_PREFIX=$HERE/src/objslampp/.anaconda3
  source $CONDA_PREFIX/bin/activate

  show-ros
  echo "PYTHONPATH: $PYTHONPATH"
  echo "PYTHON: $(command which python)"
}

HERE=${0:a:h}
setup
