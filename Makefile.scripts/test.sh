#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT

source .anaconda3/bin/activate

pip install -U pytest

CI=${CI:-false}
if [ "$CI" == "true" ]; then
  pytest -v tests -m 'not gpu and not heavy'
else
  pytest -v tests
fi
