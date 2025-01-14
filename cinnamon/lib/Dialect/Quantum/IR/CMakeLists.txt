add_mlir_dialect_library(QuantumIR
        QuantumBase.cpp
        QuantumOps.cpp
        QuantumTypes.cpp
        
    DEPENDS
        QuantumIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces

        MLIRArithDialect
        MLIRDLTIDialect
        MLIRFunctionInterfaces
        MLIRInferIntRangeInterface
        MLIRMemRefDialect
        MLIRSupport
)
