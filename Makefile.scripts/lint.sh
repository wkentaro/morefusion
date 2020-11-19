#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

cd $ROOT

source .anaconda3/bin/activate

echo_bold "==> Installing black, hacking and mypy"
pip_install -U "black==19.10b0" hacking mypy

echo_bold "==> Linting with black"
black --check .

echo_bold "==> Linting with flake8"
flake8 .

echo_bold "==> Linting with mypy"
mypy --package morefusion --ignore-missing-imports
