add_mlir_dialect_library(QuantumIR
        QuantumAttributes.cpp
        QuantumBase.cpp
        QuantumOps.cpp
        QuantumTypes.cpp
        
    DEPENDS
        QuantumIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        QuantumEnums
        QuantumTransforms
        QIRIR
)