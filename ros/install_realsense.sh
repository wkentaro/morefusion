#!/bin/bash

set -e

if ! dpkg --get-selections | egrep '^librealsense' &>/dev/null; then
  set -x

  sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
  sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo $(lsb_release -cs) main" -u

  sudo apt-get install librealsense2-dkms
  sudo apt-get install librealsense2-utils

  sudo apt-get install librealsense2-dev
  sudo apt-get install librealsense2-dbg
fi
