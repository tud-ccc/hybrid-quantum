add_mlir_conversion_library(QILLRToQuantum
        QILLRToQuantum.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        QuantumIR
        QILLRIR
)
