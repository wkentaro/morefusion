#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

CI=${1:-0}
shift

cd $ROOT

source .anaconda3/bin/activate

set -e

for file in $(find $ROOT/checks -name '*.py'); do
  echo $file
  python $file
done
