add_mlir_conversion_library(MLIRQuantumToQIR
    QuantumToQIR.cpp
    Patterns.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/Quantum/Conversion/QuantumToQIR

    DEPENDS
    QuantumConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    QIRIR
    QuantumIR
    MLIRDialectUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)
