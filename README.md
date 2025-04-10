# hybrid-quantum

This repository contains a set of MLIR dialects for hybrid quantum-classical computations.

## About The Project

The project aims to develop a comprehensive framework for hybrid quantum-classical computing that includes a collection of quantum-specific dialects, enabling a progressive lowering to targeted backends such as the Quantum Intermediate Representation (QIR) and LLVM IR. By implementing optimization passes tailored for the quantum dialect, we seek to enhance the performance and efficiency of quantum algorithms. Additionally, this framework will facilitate the integration of classical dialects, allowing for seamless collaboration between quantum and classical computing paradigms. These features are essential for improving the programmability and usability of quantum architectures, making them more accessible to researchers and developers while promoting interoperability across various quantum platforms.

## Building

Make sure to provide all dependencies required by the project, either by installing them to the system-default locations, or by setting the search location hints.

### Dependencies

The project depends on [LLVM](https://github.com/llvm/llvm-project) version `20.1.1` (`424c2d9`).

```sh
# Configure LLVM
cmake -S $LLVM_PREFIX/../llvm -B $LLVM_PREFIX \
   -G Ninja \
   -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
   -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
   -DLLVM_BUILD_TOOLS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_SHARED_LIBS=ON \
   -DLLVM_OPTIMIZED_TABLEGEN=ON

# Build LLVM
ninja -C $LLVM_PREFIX
```

As a backend it supports [QIR Runner](https://github.com/qir-alliance/qir-runner) in version `0.7.6`.
QIR runner is a Rust library providing an implementation of the QIR spec.

```sh
cargo build -Znext-lockfile-bump --release
```

### quantum-mlir

The `hybrid-quantum` project is built using CMAKE (version 3.20 or newer).

```sh
# Configure
cmake -S . -B build \
   -G Ninja \
   -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
   -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir \
   -DBACKEND_QIR=1 \
   -DQIR_DIR=$QIR_PREFIX

# Build
ninja -C build

# Tests
ninja -C build check-quantum-mlir
```

The following CMAKE variables can be configured:

| NAME | TYPE | DESCRIPTION |
| --- | --- | --- |
| LLVM_DIR  | STRING  | Path to the CMake directory of an LLVM installation, e.g. `~/tools/llvm-15/lib/cmake/llvm` |
| MLIR_DIR  | STRING  | Path to the CMake directory of an MLIR installation, e.g. `~/tools/llvm-15/lib/cmake/mlir` |
| QIR | BOOL | Set whether the QIR runner backend should be enabled. If `true` the `QIR_DIR` must be set. |
| QIR_DIR | STRING  | Path to the target directory of QIR runner, e.g. `~/tools/qir-runner/target/release` |

## License

Distributed under the BSD 3-clause "Clear" License. See `LICENSE.txt` for more information.
