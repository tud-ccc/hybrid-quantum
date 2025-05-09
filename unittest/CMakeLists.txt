################################################################################
# quantum-mlir-tests
#
# The quantum-mlir unit test project.
################################################################################

project(quantum-mlir-tests)

find_package(doctest REQUIRED)
include(doctest)

include(CTest)

add_executable(${PROJECT_NAME}
    main.cpp
    QuantumIf.cpp
)
if(APPLE)
    target_include_directories(${PROJECT_NAME}
        PRIVATE
            /opt/homebrew/include        
    )
endif()
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRIR
        MLIRSupport
        QuantumIR
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        -fno-rtti
)

doctest_discover_tests(${PROJECT_NAME})