add_mlir_conversion_library(QuantumToQIR
        QuantumToQIR.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        QuantumIR
        QIRIR
)
