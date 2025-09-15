add_mlir_dialect_library(RVSDGTransforms
        ControlFlowHoisting.cpp

    ENABLE_AGGREGATION

    DEPENDS
        RVSDGPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
