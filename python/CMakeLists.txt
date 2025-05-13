include(AddMLIRPython)

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
################################################################################

add_mlir_python_common_capi_library(MLIRPythonCAPI
  INSTALL_COMPONENT MLIRPythonModules
  INSTALL_DESTINATION python_packages/mlir_core/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    ${_ADDL_TEST_SOURCES}
)

################################################################################
# The fully assembled package of modules.
# This must come last.
################################################################################

add_mlir_python_modules(MLIRPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mlir_core/mlir"
  INSTALL_PREFIX "python_packages/mlir_core/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    ${_ADDL_TEST_SOURCES}
  COMMON_CAPI_LINK_LIBS
    MLIRPythonCAPI
)
