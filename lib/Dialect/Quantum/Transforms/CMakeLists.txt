add_mlir_dialect_library(QuantumTransforms
        Hermitian.cpp
        GateOptimization.cpp
        PhasePolynomialMerge.cpp
        VerifyLinearity.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QuantumPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        QuantumAnalysis
        QuantumInterfaces
)
