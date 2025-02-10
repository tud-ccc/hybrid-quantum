################################################################################
# quantum-opt
#
# The quantum-mlir optimizer driver.
################################################################################

project(quantum-opt)

add_executable(${PROJECT_NAME}
    quantum-opt.cpp
)

# Link all standard MLIR dialect and conversion libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIROptLib
        ${dialect_libs}
        ${conversion_libs}
)
