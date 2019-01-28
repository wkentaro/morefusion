#!/bin/bash

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_files() {
  git checkout master
  git add $*
  git commit --message "Travis build $TRAVIS_BUILD_NUMBER"
}

upload_files() {
  git remote add deploy https://${GITHUB_TOKEN}@github.com/$TRAVIS_REPO_SLUG.git
  git push deploy master
}

setup_git
commit_files $*
upload_files
