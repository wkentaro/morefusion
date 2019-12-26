#!/usr/bin/env zsh

leave () {
  unset PYTHONPATH
  unset CMAKE_PREFIX_PATH
  conda deactivate
}

leave
