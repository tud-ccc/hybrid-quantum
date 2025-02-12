#!/bin/bash

if [[ $1 != "no-qir" ]]; then

git clone https://github.com/lschuetze/qir-runner qir-runner

fi

if [[ $1 != "no-llvm" ]]; then

git clone https://github.com/oowekyala/llvm-project llvm
cd llvm
git checkout cinnamon-llvm
mkdir -p build

fi