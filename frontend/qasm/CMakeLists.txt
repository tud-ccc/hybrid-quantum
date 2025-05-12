################################################################################
# qasm-quantum-frontend
#
# The QASM to quantum-mlir Python frontend project.
################################################################################

project(qasm-quantum-frontend)

if(FRONTEND_QASM)
    # Defined in MLIRConfig.cmake
    if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
        message(FATAL_ERROR "ERROR: MLIR was not built with Python bindings enabled")
    endif()

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    set(QASM_FRONTEND_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Directory of the QASM Python frontend")

    message(STATUS "QASM frontend enabled in: " ${QASM_FRONTEND_DIR})

endif()
