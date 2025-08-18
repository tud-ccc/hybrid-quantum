add_mlir_dialect_library(QILLRTransforms
        DecomposeUGates.cpp
        ReplaceRepeatedMeasurementReads.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QILLRPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
