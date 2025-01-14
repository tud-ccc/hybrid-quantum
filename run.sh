#!/bin/bash

export PATH=$(pwd)/llvm/build/bin:$PATH
cd cinnamon 
llvm_prefix=../llvm/build

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Build directory does not exist. Running cmake..."
    cmake -S . -B "build" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_DIR="$llvm_prefix"/lib/cmake/llvm \
        -DMLIR_DIR="$llvm_prefix"/lib/cmake/mlir \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_LINKER_TYPE=DEFAULT \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
fi

# Change to build directory and run ninja
cd build && ninja 