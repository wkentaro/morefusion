#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

if [ ! -d $ROOT/.anaconda3 ]; then
  cd $ROOT
  curl -L https://github.com/wkentaro/dotfiles/raw/master/local/bin/install_anaconda3.sh | bash -s .
fi
