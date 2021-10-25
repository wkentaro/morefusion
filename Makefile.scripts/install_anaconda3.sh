#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

INSTALL_DIR=$ROOT

if [ -e $INSTALL_DIR/.anaconda3 ]; then
  echo_bold "==> Anaconda3 is already installed: $INSTALL_DIR/.anaconda3"
  exit 0
fi

echo_bold "==> Installing Anaconda3 to: $INSTALL_DIR/.anaconda3"

TMPDIR=$(mktemp -d)
cd $TMPDIR

if [ "$(uname)" = "Linux" ]; then
  URL='https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh'
elif [ "$(uname)" = "Darwin" ]; then
  URL='https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh'
else
  echo_warning "==> Unsupported platform: $(uname)"
  exit 1
fi

if which wget &>/dev/null; then
  wget -q $URL -O miniconda3.sh
else
  curl -s -L $URL -o miniconda3.sh
fi

unset PYTHONPATH

bash ./miniconda3.sh -p $INSTALL_DIR/.anaconda3 -b
cd -
rm -rf $TMPDIR

source $INSTALL_DIR/.anaconda3/bin/activate
conda_install 'python<3.9'  # for open3d-python
