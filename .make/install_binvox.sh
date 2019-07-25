#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

OUTPUT=$ROOT/.anaconda3/bin/binvox
if [ ! -f $OUTPUT ]; then
  if [ "$(uname)" = "Darwin" ]; then
    URL=http://www.patrickmin.com/binvox/mac/binvox
  else
    URL=http://www.patrickmin.com/binvox/linux64/binvox
  fi
  if which wget &>/dev/null; then
    wget -q $URL -O $OUTPUT
  else
    curl -s $URL -o $OUTPUT
  fi
  chmod u+x $OUTPUT
fi

OUTPUT=$ROOT/.anaconda3/bin/viewvox
if [ ! -f $OUTPUT ]; then
  if [ "$(uname)" = "Darwin" ]; then
    URL=http://www.patrickmin.com/viewvox/mac/viewvox
  else
    URL=http://www.patrickmin.com/viewvox/linux64/viewvox
  fi
  if which wget &>/dev/null; then
    wget -q $URL -O $OUTPUT
  else
    curl -s $URL -o $OUTPUT
  fi
  chmod u+x $OUTPUT
fi

source $ROOT/.anaconda3/bin/activate
OUTPUT=$(python -c 'import site, sys; sys.stdout.write(site.getsitepackages()[0])')/binvox_rw.py
if [ ! -f $OUTPUT ]; then
  URL=https://raw.githubusercontent.com/dimatura/binvox-rw-py/public/binvox_rw.py
  if which wget &>/dev/null; then
    wget -q $URL -O $OUTPUT
  else
    curl -s $URL -o $OUTPUT
  fi
fi
