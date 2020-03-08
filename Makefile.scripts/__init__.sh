#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
ROOT=$(realpath $HERE/..)

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

echo_warning () {
  echo -e "\033[33m$*\033[0m"
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

pip_install () {
  pip install --progress-bar off $@
}

conda_install () {
  conda install -y -q $@
}

git_clone () {
  git clone -q $@
}
