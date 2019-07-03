#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

bin_file=$ROOT/.anaconda3/bin/binvox
if [ ! -f $bin_file ]; then
  if [ "$(uname)" = "Darwin" ]; then
    wget -q http://www.patrickmin.com/binvox/mac/binvox -O $bin_file
  else
    wget -q http://www.patrickmin.com/binvox/linux64/binvox -O $bin_file
  fi
  chmod u+x $bin_file
fi

bin_file=$ROOT/.anaconda3/bin/viewvox
if [ ! -f $bin_file ]; then
  if [ "$(uname)" = "Darwin" ]; then
    wget -q http://www.patrickmin.com/viewvox/mac/viewvox -O $bin_file
  else
    wget -q http://www.patrickmin.com/viewvox/linux64/viewvox -O $bin_file
  fi
  chmod u+x $bin_file
fi

source $ROOT/.anaconda3/bin/activate
out_file=$(python -c 'import site, sys; sys.stdout.write(site.getsitepackages()[0])')/binvox_rw.py
if [ ! -f $out_file ]; then
  wget https://raw.githubusercontent.com/dimatura/binvox-rw-py/public/binvox_rw.py -O $out_file
fi
