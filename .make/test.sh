#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

GPU=${1:-1}

cd $ROOT

source .anaconda3/bin/activate

pip install -U pytest

if [ "$GPU" = "1" ]; then
  pytest -v tests
else
  pytest -v tests -m 'not gpu'
fi
