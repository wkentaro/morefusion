#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT/src

if [ ! -e v-hacd ]; then
  git clone https://github.com/kmammou/v-hacd.git
fi

if [ ! -e $ROOT/.anaconda3/bin/testVHACD ]; then
  cd $ROOT/src/v-hacd
  mkdir -p build
  cd build
  cmake ../src
  make -j

  cd $ROOT/.anaconda3/bin
  ln -s ../../src/v-hacd/build/test/testVHACD
fi
