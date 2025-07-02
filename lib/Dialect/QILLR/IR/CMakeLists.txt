add_mlir_dialect_library(QILLRIR
        QILLRAttributes.cpp
        QILLRBase.cpp
        QILLROps.cpp
        QILLRTypes.cpp

    DEPENDS
        QILLRIncGen

    LINK_LIBS PUBLIC
        MLIRCallInterfaces
        MLIRFunctionInterfaces
        MLIRInferTypeOpInterface
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRControlFlowInterfaces
)
