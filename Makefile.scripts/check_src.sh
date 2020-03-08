#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

echo_bold "==> Checking the remaining change in src/"
for dir in src/*; do
  if [ ! -d $dir/.git ]; then
    continue
  fi
  diff=$(cd $dir && git diff HEAD)
  if [ "$diff" != "" ]; then
    echo_error "==> Found a diff in the source: $dir"
    echo "$diff"
    exit 1
  fi
done
