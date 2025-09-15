add_mlir_dialect_library(QuantumTransforms
        Hermitian.cpp
        GateOptimization.cpp
        MultiQubitLegalization.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QuantumPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
