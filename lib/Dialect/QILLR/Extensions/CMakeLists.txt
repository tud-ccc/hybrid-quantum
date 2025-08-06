set(LLVM_OPTIONAL_SOURCES
    InlinerExtension.cpp
    )

add_mlir_extension_library(QILLRInlinerExtension
    InlinerExtension.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/quantum-mlir/Dialect/QILLR/Extensions

    LINK_LIBS PUBLIC
    MLIRIR
    QILLRIR
    )
