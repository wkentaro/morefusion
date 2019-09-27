#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

echo_bold "==> Installing OpenNI2"

mkdir -p $ROOT/src
cd $ROOT/src

if [ ! -d OpenNI2 ]; then
  git clone https://github.com/occipital/OpenNI2.git
  cd OpenNI2
  git checkout b333a3b512fee95607f117d858bccb4e9ba55699
else
  cd OpenNI2
fi

make -s
