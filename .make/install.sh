#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

echo_warning () {
  echo -e "\033[1;33m$*\033[0m"
}

echo_error () {
  echo -e "\033[1;31m$*\033[0m"
}

conda_check_installed () {
  if [ ! $# -eq 1 ]; then
    echo "usage: $0 PACKAGE_NAME"
    return 1
  fi
  conda list | awk '{print $1}' | egrep "^$1$" &>/dev/null
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT

source .anaconda3/bin/activate

# ---------------------------------------------------------------------------------------

# trimesh dependency
conda_check_installed libspatialindex || conda install libspatialindex -y

# pygraphviz
conda_check_installed pygraphviz || conda install -c anaconda pygraphviz -y

if [ "$CI" = "true" ]; then
  # it fails with following error with pip install:
  # ImportError: dlopen: cannot load any more object with static TLS
  conda install -y scikit-learn
fi

echo_bold "==> Installing latest pip and setuptools"
pip install -U pip setuptools wheel

echo_bold "==> Installing cython and numpy"
pip install cython numpy

echo_bold "==> Checking the remaining change in src/"
for dir in src/*; do
  if [ ! -d $dir/.git ]; then
    continue
  fi
  diff=$(cd $dir && git diff)
  if [ "$diff" != "" ]; then
    echo_error "==> Found a diff in the source: $dir"
    echo "$diff"
    exit 1
  fi
done

echo_bold "==> Installing with requirements-dev.txt"
pip install -r requirements-dev.txt

pip install -e .

# ---------------------------------------------------------------------------------------

echo_bold "\nAll is well! You can start using this!

  $ source .anaconda3/bin/activate
  $ python checks/datasets_checks/ycb_video_checks/check_models.py
"
