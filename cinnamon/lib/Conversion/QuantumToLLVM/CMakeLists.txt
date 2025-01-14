add_mlir_conversion_library(MLIRQuantumToLLVM
    QuantumToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/Quantum/Conversion/QuantumToLLVM

    DEPENDS
    QuantumConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    QuantumIR
    MLIRDialectUtils
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
)
