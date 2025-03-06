################################################################################
# cquantum-translate
#
# The quantum translation driver.
################################################################################

project(quantum-translate)

add_executable(${PROJECT_NAME}
    quantum-translate.cpp
)

# Link all standard MLIR dialect and translation libs.
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslateLib
        MLIRSupport
        ${dialect_libs}
        ${translation_libs}
)