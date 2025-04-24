# hybrid-quantum

This repository contains a set of MLIR dialects for hybrid quantum-classical computations.

## About The Project

The project aims to develop a comprehensive framework for hybrid quantum-classical computing that includes a collection of quantum-specific dialects, enabling a progressive lowering to targeted backends such as the Quantum Intermediate Representation (QIR) and LLVM IR. By implementing optimization passes tailored for the quantum dialect, we seek to enhance the performance and efficiency of quantum algorithms. Additionally, this framework will facilitate the integration of classical dialects, allowing for seamless collaboration between quantum and classical computing paradigms. These features are essential for improving the programmability and usability of quantum architectures, making them more accessible to researchers and developers while promoting interoperability across various quantum platforms.

## Building

Make sure to provide all dependencies required by the project, either by installing them to the system-default locations, or by setting the search location hints.

### Development

For the development of the project we pinned required Python packages in the `requirements.txt` file in the base project folder.
To install the dependency you need `python3`.
We recommend to install the dependencies via a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), for example in `bash/zsh`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

To increase the quality of the code and rejections from CI verifiers please install the `pre-commit` hooks that we provide.
`pre-commit` will be installed into the virtual environment.
```sh
pre-commit install
```

### Dependencies

The project depends on [LLVM](https://github.com/llvm/llvm-project) version `20.1.1` (`424c2d9`).
You have to set `MLIR_ENABLE_BINDINGS_PYTHON` to build [MLIR Python bindings](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Bindings/Python.md) if you want to use our Python frontend.

```sh
# Setup the virtual environment for the MLIR Python bindings dependencies
python3 -m venv .venv
source .venv/bin/activate

# It is recommended to upgrade pip:
python3 -m pip install --upgrade pip

# Install the Python bindings requirements
python3 -m pip install -r $LLVM_PREFIX/mlir/python/requirements.txt

# Configure LLVM
cmake -S $LLVM_PREFIX/llvm -B $LLVM_PREFIX/build \
   -G Ninja \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DPython3_EXECUTABLE="$VENV_DIR/bin/python3" \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
   -DLLVM_BUILD_TOOLS=ON \
   -DBUILD_SHARED_LIBS=ON \
   -DLLVM_OPTIMIZED_TABLEGEN=ON

# Build LLVM
ninja -C $LLVM_PREFIX/build
```

We provide a [Qiskit](https://www.ibm.com/quantum/ecosystem) `OpenQASM` frontend tested with Qiskit version `2.0.0`.
To use the frontend You can install the dependency to your virtual environment
```sh
python3 -m pip install -r ./frontend/qasm/requirements.txt
```

As a backend it supports [QIR Runner](https://github.com/qir-alliance/qir-runner) in version `0.7.6`.
QIR runner is a Rust library providing an implementation of the QIR spec.

```sh
cargo build -Znext-lockfile-bump --release
```

### quantum-mlir

The `hybrid-quantum` project is built using CMAKE (version 3.22 or newer).

```sh

# Install the frontend dependencies to the virtual environment
python3 -m pip install -r frontend/requirements.txt

# Configure
cmake -S . -B build \
   -G Ninja \
   -DLLVM_DIR=$LLVM_PREFIX/build/lib/cmake/llvm \
   -DMLIR_DIR=$LLVM_PREFIX/build/lib/cmake/mlir \
   -DPython3_EXECUTABLE="$VENV_DIR/bin/python3" \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DBACKEND_QIR=ON \
   -DQIR_DIR=$QIR_PREFIX \
   -DFRONTEND_QASM=ON

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
| BACKEND_QIR | BOOL | Set whether the QIR runner backend should be enabled. If `ON` the `QIR_DIR` must be set. |
| QIR_DIR | STRING  | Path to the target directory of QIR runner, e.g. `~/tools/qir-runner/target/release` |
| FRONTEND_QASM | BOOL | Set whether the Qiskit OpenQASM frontend should be enabled. If `ON` MLIR must be built with `MLIR_ENABLE_BINDINGS_PYTHON` must be set. |

## License

Distributed under the BSD 3-clause "Clear" License. See `LICENSE.txt` for more information.
