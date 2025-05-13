if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(AddMLIRPython)

################################################################################
# Pure python sources and generated code
################################################################################

################################################################################
# Structural groupings.
################################################################################

declare_mlir_python_sources(QuantumPythonSources)
declare_mlir_python_sources(QuantumPythonSources.Dialects
  ADD_TO_PARENT QuantumPythonSources)

################################################################################
# Pure python sources and generated code
################################################################################

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/QIROps.td
  SOURCES
    dialects/qir.py
  DIALECT_NAME qir
)

################################################################################
# Python extensions.
# The sources for these are all in lib/Bindings/Python, but since they have to
# be rebuilt for each package and integrate with the source setup here, we
# just reference them here instead of having ordered, cross package target
# dependencies.
################################################################################

set(PYTHON_SOURCE_DIR "${CMAKE_SOURCE_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(MLIRPythonExtension.Dialects.QIR.Pybind
  MODULE_NAME _mlirDialectsQIR
  ADD_TO_PARENT MLIRPythonSources.Dialects.qir
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    DialectQIR.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIQIR
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
#
# The presence of this aggregate library is part of the long term plan, but its
# use needs to be made more flexible.
#
# TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
# once ready.
################################################################################

add_mlir_python_common_capi_library(MLIRPythonCAPI
  INSTALL_COMPONENT MLIRPythonModules
  INSTALL_DESTINATION python_packages/mlir_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/mlir_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
)

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

add_mlir_python_modules(MLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/mlir_core/mlir"
  INSTALL_PREFIX "python_packages/mlir_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    MLIRPythonCAPI
)
