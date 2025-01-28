################################################################################
# check-cinm-mlir
#
# The cinm-mlir regression test project.
################################################################################
################################################################################
# check-cinm-mlir
#
# The cinm-mlir regression test project.
################################################################################

project(check-cinm-mlir)

# Create the test suite.
add_lit_testsuite(${PROJECT_NAME}
    "Running the cinm-mlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${TEST_DEPENDS}
)
add_lit_testsuites(cinmMLIR
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
    ${TEST_DEPENDS}
)