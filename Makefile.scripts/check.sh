#!/bin/bash

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

CI=${1:-0}
shift

cd $ROOT

source .anaconda3/bin/activate

set -e

for file in $(find $ROOT/checks -name '*.py' | sort); do
  echo_bold "==> $file"
  python $file
done
