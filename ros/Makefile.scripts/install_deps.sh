#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE:-$0}))
ROOT=$(realpath $HERE/../..)

CATKIN_WS=$ROOT/ros

set -e

source $ROOT/.anaconda3/bin/activate

set -x

# for roslaunch
pip install rospkg

# for jsk_recognition_utils
pip install fcn

{ set +x; } 2>/dev/null

source /opt/ros/noetic/setup.bash

set -x

cd $CATKIN_WS

rosdep install --from-path src -r -y
