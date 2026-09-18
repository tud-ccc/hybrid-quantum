add_mlir_library(QuantumAnalysis
    RegisterRangesAnalysis.cpp
    PhasePolynomialAnalysis.cpp
    LinearUseAnalysis.cpp

    ENABLE_AGGREGATION
    DEPENDS
        QuantumIncGen
)
