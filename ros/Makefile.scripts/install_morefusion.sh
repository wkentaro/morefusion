#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE:-$0}))
ROOT=$(realpath $HERE/../..)

CATKIN_WS=$ROOT/ros

set -e

source $ROOT/.anaconda3/bin/activate

set -x

cd $CATKIN_WS/src/morefusion
pip install -e .

{ set +x; } 2>/dev/null

conda deactivate

source /opt/ros/noetic/setup.bash

set -x

catkin build morefusion_panda morefusion_panda_ycb_video
