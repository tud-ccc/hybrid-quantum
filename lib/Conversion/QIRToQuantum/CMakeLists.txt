add_mlir_conversion_library(QIRToQuantum
        QIRToQuantum.cpp

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
