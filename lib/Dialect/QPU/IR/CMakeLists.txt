add_mlir_dialect_library(QPUIR
        QPUAttributes.cpp
        QPUBase.cpp
        QPUOps.cpp
        QPUTypes.cpp
        
    DEPENDS
        QPUIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        QPUTransforms
        QuantumIR
        MLIRArithDialect
        MLIRArithTransforms
)
