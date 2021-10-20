#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE:-$0}))
ROOT=$(realpath $HERE/../..)

CATKIN_WS=$ROOT/ros

set -e

source /opt/ros/noetic/setup.bash

set -x

cd $CATKIN_WS

catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DOCTOMAP_OMP=1

(cd src && wstool update)
