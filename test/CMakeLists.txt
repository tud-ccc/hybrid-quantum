################################################################################
# check-quantum-mlir
#
# The quantum-mlir regression test project.
################################################################################

project(check-quantum-mlir)

if (${BACKEND_QIR})
    # Explicitly set the variable for substitution
    set(QIR_SHLIBS "${QIR_DIR}/libqir_backend.so,${QIR_DIR}/libqir_stdlib.so")

    message(STATUS "Using backend QIR libs in: ${QIR_DIR}")
endif()

# Configure the testing site configuration.
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
    FileCheck count not
    mlir-runner
    quantum-opt
)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the quantum-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(QUANTUMMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)