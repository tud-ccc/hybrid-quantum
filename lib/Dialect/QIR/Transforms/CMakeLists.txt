add_mlir_dialect_library(QIRTransforms
        DecomposeUGates.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QIRPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
