add_mlir_dialect_library(QPUTransforms
    TranspileCircuit.cpp
    SabreSwaps.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QPUPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        QPUAnalysis
)
