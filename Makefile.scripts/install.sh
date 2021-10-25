#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

cd $ROOT

source .anaconda3/bin/activate

# ---------------------------------------------------------------------------------------

# trimesh dependency
conda_check_installed libspatialindex || conda_install libspatialindex

# pygraphviz
conda_check_installed pygraphviz || conda_install pygraphviz

if [ "$CI" = "true" ]; then
  # it fails with following error with pip install:
  # ImportError: dlopen: cannot load any more object with static TLS
  conda_install -y scikit-learn
fi

echo_bold "==> Installing latest pip and setuptools"
pip_install -U pip setuptools wheel

echo_bold "==> Installing cython and numpy"
pip_install cython numpy

echo_bold "==> Installing with requirements-dev.txt"
pip_install -r requirements-dev.txt

echo_bold "==> Installing pre-commit"
pre-commit install

echo_bold "==> Installing main package"
pip_install -e .  --no-deps

echo_bold "==> Checking the availability of Cupy"
if ! python -c 'import cupy' &>/dev/null; then
  echo_warning "Cupy is not yet installed. Please install it manually e.g., pip install cupy."
fi

# ---------------------------------------------------------------------------------------

echo_bold "\nAll is well! You can start using this!

  $ source .anaconda3/bin/activate
  $ python checks/datasets_checks/ycb_video_checks/check_models.py
"
