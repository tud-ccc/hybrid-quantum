add_mlir_conversion_library(ScfToRVSDG
        ScfToRVSDG.cpp

    DEPENDS
        ConversionIncGen

    LINK_COMPONENTS
        Core

    LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRTransformUtils
        MLIRIR
        QILLRIR
        QuantumIR
        RVSDGIR
)
