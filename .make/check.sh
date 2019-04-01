#!/bin/bash

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

CI=${1:-0}
shift

cd $ROOT

source .anaconda3/bin/activate

set -e

for file in $(find $ROOT/checks -name '*.py' | sort); do
  echo_bold "==> $file"
  python $file
done
