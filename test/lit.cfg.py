# -*- Python -*-

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "quantum-mlir"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.quantum_obj_root, "test")


# Searches for a runtime library with the given name and returns the found path.
# Correctly handles the platforms shared library directory and naming conventions.
def find_runtime(dir, name):
    path = ""
    for prefix in ["", "lib"]:
        path = os.path.join(dir, f"{prefix}{name}{config.llvm_shlib_ext}")
        if os.path.isfile(path):
            break
    return path


# Searches for a runtime library with the given name and returns a tool
# substitution of the same name and the found path.
def add_runtime(name):
    return ToolSubst(f"%{name}", find_runtime(config.llvm_lib_dir, name))


# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "README.md", "LICENSE.txt"]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%llvm_src_root", config.llvm_src_root))

config.substitutions.append(("%qir_shlibs", config.qir_shlibs))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
config.quantum_tools_dir = os.path.join(config.quantum_obj_root, "bin")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.quantum_tools_dir, append_path=True)

tool_dirs = [config.quantum_tools_dir, config.llvm_tools_dir]
tools = [
    "quantum-opt",
    "mlir-runner",
    add_runtime("mlir_runner_utils"),
    add_runtime("mlir_c_runner_utils"),
]

# Find QASM frontend
qasm_import = config.qasm_frontend_dir + "/QASM2MLIR.py"
# Optional tools
tools.extend(
    [
        ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
        ToolSubst("qasm-import", qasm_import, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)
