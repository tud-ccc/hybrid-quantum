add_mlir_library(QPUAnalysis
        SabreSwapAnalysis.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QPUIncGen
)
