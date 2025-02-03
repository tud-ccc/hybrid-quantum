add_mlir_conversion_library(MLIRQIRToLLVM
    QIRToLLVM.cpp
    Patterns.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/QIR/Conversion/QIRToLLVM

    DEPENDS
    QIRConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    QIRIR
    MLIRDialectUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)
