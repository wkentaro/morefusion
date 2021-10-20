#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE:-$0}))
ROOT=$(realpath $HERE/../..)

CATKIN_WS=$ROOT/ros

set -e

source $ROOT/.anaconda3/bin/activate
conda deactivate

source /opt/ros/noetic/setup.bash

set -x

cd $CATKIN_WS

catkin build
