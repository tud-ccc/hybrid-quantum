################################################################################
# quantum-mlir
#
# MLIR dialect implementing hybrid quantum-classical representations.
################################################################################

cmake_minimum_required(VERSION 3.23)

project(quantum-mlir
    VERSION     0.1.0
    LANGUAGES   C CXX
)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project options
option(BACKEND_QIR "Use QIR runner backend" ON)
option(FRONTEND_QASM "Use Qiskit OpenQASM frontend" ON)

# Project-specific CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ---------------------------------------------------------------------------
# LLVM / MLIR
# ---------------------------------------------------------------------------

find_package(MLIR 23.1.1 EXACT REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

if(POLICY CMP0219)
  cmake_policy(SET CMP0219 NEW)
endif()
include(HandleLLVMOptions)

# TableGen and project-wide include paths
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories("${PROJECT_BINARY_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include")

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------

add_compile_options(
  -Wno-ambiguous-reversed-operator
  $<$<CXX_COMPILER_ID:Clang>:-fdebug-default-version=4>
)

# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(lib/CAPI)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")

  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  if(NOT MLIR_PYTHON_PACKAGE_PREFIX)
    set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_quantum" CACHE STRING "" FORCE)
  endif()
  if(NOT MLIR_BINDINGS_PYTHON_NB_DOMAIN)
    set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir_quantum" CACHE STRING "" FORCE)
  endif()
  if(NOT MLIR_BINDINGS_PYTHON_INSTALL_PREFIX)
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/quantum/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)
  endif()

  add_subdirectory(python)
endif()

add_subdirectory(frontend)
add_subdirectory(tools)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

add_subdirectory(test)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(unittest)
endif()
