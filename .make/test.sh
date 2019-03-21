#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

CI=${1:-0}
shift

cd $ROOT

source .anaconda3/bin/activate

pip install -U pytest

if [ "$CI" == "0" ]; then
  pytest -v tests
else
  test "$CI" == "1" || exit 1
  pytest -v tests -m 'not gpu and not heavy'
fi
