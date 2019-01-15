#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

echo_warning () {
  echo -e "\033[33m$*\033[0m"
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

echo_bold "==> Installing latest pip and setuptools"
pip install -U pip setuptools wheel

echo_bold "==> Installing dev tool like flake8, pytest and ipython"
pip install -U flake8 pytest ipython ipdb pycd

# ---------------------------------------------------------------------------------------

echo_bold "==> Installing python modules"
pip install -r requirements.txt

# trimesh dependency
conda_check_installed libspatialindex || conda install libspatialindex -y

pip install -e .

# ---------------------------------------------------------------------------------------

echo_bold "\nAll is well! You can start using this!

  $ source .anaconda3/bin/activate
"
