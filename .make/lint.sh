#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT

source .anaconda3/bin/activate

echo_bold "==> Installing flake8 and mypy"
pip install -U flake8 mypy

echo_bold "==> Linting with flake8"
flake8 .

echo_bold "==> Linting with mypy"
mypy -p objslampp --ignore-missing-imports
