#!/bin/bash -e

HERE=$(realpath $(dirname ${BASH_SOURCE[0]}))
source $HERE/__init__.sh
ROOT=$(realpath $HERE/..)

cd $ROOT

source .anaconda3/bin/activate

pip_install black==19.10b0
echo_bold "==> Linting with black"
black --check .

pip install -U hacking
echo_bold "==> Linting with flake8"
flake8 .

pip install -U mypy types-termcolor types-six types-PyYAML types-frozendict
echo_bold "==> Linting with mypy"
mypy --package morefusion --ignore-missing-imports
