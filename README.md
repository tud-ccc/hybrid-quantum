# hybrid-quantum

This repository contains a set of MLIR dialects for hybrid quantum-classical computations.

## About The Project

The project aims to develop a comprehensive framework for hybrid quantum-classical computing that includes a collection of quantum-specific dialects, enabling a progressive lowering to targeted backends such as the Quantum Intermediate Representation (QIR) and LLVM IR. By implementing optimization passes tailored for the quantum dialect, we seek to enhance the performance and efficiency of quantum algorithms. Additionally, this framework will facilitate the integration of classical dialects, allowing for seamless collaboration between quantum and classical computing paradigms. These features are essential for improving the programmability and usability of quantum architectures, making them more accessible to researchers and developers while promoting interoperability across various quantum platforms.

## Building

The `hybrid-quantum` project is built using CMAKE (version 3.20 or newer).
Make sure to provide all dependencies required by the project, either by installing them to the system-default locations, or by setting the search location hints.

The project depends on a [patched version](https://github.com/oowekyala/llvm-project) of `LLVM 18.1.6` (`6f89431c3d4de87df6d76cf7ffa73bfa881607b7`).
As a backend it supports [QIR Runner](https://github.com/qir-alliance/qir-runner) in version `0.7.5`.

```sh
# Configure
cmake -S . -B build \
   -G Ninja \
   -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
   -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir \
   -DQIR \
   -DQIR_DIR=$QIR_PREFIX/target

# Build
cmake --build build
```

The following CMAKE variables can be configured:

| NAME | TYPE | DESCRIPTION |
| --- | --- | --- |
| LLVM_DIR  | STRING  | Path to the CMake directory of an LLVM installation, e.g. `~/tools/llvm-15/lib/cmake/llvm` |
| MLIR_DIR  | STRING  | Path to the CMake directory of an MLIR installation, e.g. `~/tools/llvm-15/lib/cmake/mlir` |
| QIR | BOOL | Set whether the QIR runner backend should be enabled. If set `QIR_SOURCE_PATH` must be set. |
| QIR_DIR | STRING  | Path to the SOURCE directory of QIR runner, e.g. `~/tools/qir-runner` |

## License

Distributed under the BSD 3-clause "Clear" License. See `LICENSE.txt` for more information.

## Contributors

* Lars Schütze (<lars.schuetze@tu-dresden.de>)
* Washim S. Neupane (<washim_sharma.neupane@mailbox.tu-dresden.de>)
