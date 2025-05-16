if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(QIRPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QIRPythonSources
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

declare_mlir_python_extension(QIRPythonSources.NanobindExtension
  MODULE_NAME _mlirDialectsQIR
  ADD_TO_PARENT QIRPythonSources
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
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(QIRPythonCAPI
  INSTALL_COMPONENT QIRPythonModules
  INSTALL_DESTINATION python_packages/quantum-mlir/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/quantum-mlir/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(QIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/quantum-mlir/mlir"
  INSTALL_PREFIX "python_packages/quantum-mlir/mlir"
  DECLARED_SOURCES
    QIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QIRPythonCAPI
)
