# Build recipes for this project.
#

# Load environment vars from .env file
# Write LLVM_BUILD_DIR="path" into that file or set this env var in your shell.
set dotenv-load := true

# Make sure your LLVM is tags/llvmorg-18.1.6

llvm_prefix := env_var("LLVM_BUILD_DIR")
qir_prefix := env_var("QIRRUNNER_BUILD_DIR")
build_type := env_var_or_default("LLVM_BUILD_TYPE", "RelWithDebInfo")
linker := env_var_or_default("CMAKE_LINKER_TYPE", "DEFAULT")
build_dir := "quantum/build"

llvm *ARGS:
    cmake -S {{llvm_prefix}}/../llvm -B {{llvm_prefix}} \
        -G "Ninja" \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DLLVM_BUILD_TOOLS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON

    ninja -C {{llvm_prefix}}
    ninja -C {{llvm_prefix}} llc
    ninja -C {{llvm_prefix}} opt

qir-runner *ARGS:
    cargo build -Znext-lockfile-bump --manifest-path qir-runner/backend/Cargo.toml

# execute cmake -- this is only needed on the first build
cmake *ARGS:
    cmake -S quantum -B {{build_dir}} \
        -G Ninja \
        -DCMAKE_BUILD_TYPE={{build_type}} \
        -DLLVM_DIR="{{llvm_prefix}}/lib/cmake/llvm" \
        -DMLIR_DIR="{{llvm_prefix}}/lib/cmake/mlir" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_LINKER_TYPE={{linker}} \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_USING_LINKER_mold=-fuse-ld=mold \
        -DCMAKE_CXX_USING_LINKER_mold=-fuse-ld=mold \
        {{ARGS}}

# execute a specific ninja target
doNinja *ARGS:
    ninja -C {{build_dir}} {{ARGS}}

# run build --first build needs cmake though
build: doNinja

cleanBuild:
    rm -rf {{build_dir}}
    just cmake
    just build

# run tests
test: (doNinja "check-quantum-mlir")

[no-cd]
quantum-opt *ARGS: (doNinja "quantum-opt")
    {{source_directory()}}/{{build_dir}}/bin/quantum-opt {{ARGS}}

quantum-opt-help: (quantum-opt "--help")

# Start a gdb session on quantum-opt.
debug-quantum-opt *ARGS:
    gdb --args {{build_dir}}/bin/quantum-opt {{ARGS}}

qir *ARGS:
    {{llvm_prefix}}/bin/clang -fuse-ld=mold \
        -L /lib/rustlib/x86_64-unknown-linux-gnu/lib \
        -L {{qir_prefix}} -L {{qir_prefix}}/deps \
        -lqir_backend -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc \
        {{ARGS}} -o {{replace(ARGS, ".ll", ".out")}}

# Invoke the LLVM IR compiler.
llc *ARGS:
    {{llvm_prefix}}/bin/llc {{ARGS}}

addNewDialect DIALECT_NAME DIALECT_NS:
    just --justfile ./dialectTemplate/justfile applyTemplate {{DIALECT_NAME}} {{DIALECT_NS}} "cinm-mlir" {{justfile_directory()}}
