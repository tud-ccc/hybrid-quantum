add_mlir_conversion_library(QILLRToLLVM
        QILLRToLLVM.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRLLVMDialect
        QILLRIR
        MLIRIR
)
