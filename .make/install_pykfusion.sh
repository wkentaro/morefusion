#!/bin/bash

set -e

echo_warning () {
  echo -e "\033[1;33m$*\033[0m"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

if ! which nvcc &>/dev/null; then
  echo_warning "==> nvcc not found, so skipping the installation of pykfusion"
  exit 0
fi

cd $ROOT
mkdir -p src
cd src

if [ ! -d $ROOT/.anaconda3/include/TooN ]; then
  if [ ! -d TooN ]; then
    curl -L -O https://github.com/edrosten/TooN/archive/TOON_3.2.zip
    unzip TOON_3.2.zip
    mv TooN-TOON_3.2 TooN
  fi
  (
    cd TooN
    ./configure --prefix=$ROOT/.anaconda3
    make install
  )
fi

sitepackage=$(python -c 'import site, sys; sys.stdout.write(site.getsitepackages()[0])')
link_file=$(python -c "import sys, platform; sys.stdout.write(f'pykfusion.{platform.python_implementation().lower()}-{platform.python_version()[0]}{platform.python_version()[2]}m-{platform.machine()}-{platform.system().lower()}-gnu.so')")
if [ ! -f $sitepackage/$link_file ]; then
  if [ ! -d fastfusionpp ]; then
    if [ ! -z $GITHUB_TOKEN ]; then
      git clone https://$GITHUB_TOKEN@github.com/SajadSaeediG/fastfusionpp.git
    else
      git clone https://github.com/SajadSaeediG/fastfusionpp.git
    fi
  fi
  cd fastfusionpp/pykfusion
  mkdir -p build
  cd build
  cmake ..
  make -j
  cp $link_file $sitepackage
fi
