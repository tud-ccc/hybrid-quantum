add_mlir_conversion_library(QuantumToQILLR
    QuantumToQILLR.cpp

    DEPENDS
        ConversionIncGen

    LINK_LIBS PUBLIC
        MLIRFuncTransforms
        MLIRSCFTransforms
        MLIRTensorDialect
        MLIRTransformUtils

        QuantumIR
        QILLRIR
        RVSDGIR
        QuantumAnalysis
        QuantumInterfaces
)
