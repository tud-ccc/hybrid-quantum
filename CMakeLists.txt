################################################################################
# quantum-mlir
#
# MLIR dialect implementing hybrid quantum-classical representations.
################################################################################

cmake_minimum_required(VERSION 3.22)

# Add custom modules to the search path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(quantum-mlir
    VERSION     0.1.0
    LANGUAGES   C CXX
)

# Set global language standard to C++20.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

# Set backend variables
option(BACKEND_QIR "Use QIR runner backend" ON)
# Set frontend variables
option(FRONTEND_QASM "Use Qiskit OpenQASM frontend" ON)

# Detect if this is a stand-alone build.
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    # Set shared output directories.
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)

    # Documentation will be output here.
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
endif()

# Fix for MLIR:
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
# Find MLIR.
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if(FRONTEND_QASM)
    # Defined in MLIRConfig.cmake
    if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
        message(FATAL_ERROR "ERROR: MLIR was not built with Python bindings enabled")
    endif()

    # Python virtual environment setup
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Set path to requirements.txt
    set(MLIR_REQUIREMENTS_FILE "${LLVM_BUILD_MAIN_SRC_DIR}/../mlir/python/requirements.txt")
    set(FRONTEND_REQ_FILE "${CMAKE_SOURCE_DIR}/frontend/requirements.txt")

    # Set virtualenv path
    set(PYTHON_VENV_DIR "${CMAKE_SOURCE_DIR}/.venv" CACHE STRING "venv directory")
    set(PYTHON_VENV_BIN "${PYTHON_VENV_DIR}/bin" CACHE STRING "Binary directory of Python venv")

    # Create virtual environment if it doesn't exist
    if(NOT EXISTS "${PYTHON_VENV_BIN}/activate")
        message(STATUS "Creating Python virtualenv at ${PYTHON_VENV_DIR}")
        execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${PYTHON_VENV_DIR}")
    endif()

    # Upgrade pip and install requirements
    execute_process(COMMAND "${PYTHON_VENV_BIN}/python" -m pip install --upgrade pip)
    execute_process(COMMAND "${PYTHON_VENV_BIN}/pip" install -r "${MLIR_REQUIREMENTS_FILE}")
    execute_process(COMMAND "${PYTHON_VENV_BIN}/pip" install -r "${FRONTEND_REQ_FILE}")

    # Rebind Python3 to venv
    unset(Python3_EXECUTABLE)
    set(ENV{VIRTUAL_ENV} "${PYTHON_VENV_PATH}")
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

# Load LLVM and MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Apply target properties for LLVM/MLIR globally.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Apply target properties for this project globally.
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_compile_options(
    # NOTE: C++20 warning adjustments for this project.
    -Wall
    -Wextra
    -Wno-ambiguous-reversed-operator
    # NOTE: GDB < 10.x can't handle DWARFv5 correctly.
    $<$<CXX_COMPILER_ID:Clang>:-fdebug-default-version=4>
)

add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
enable_testing()