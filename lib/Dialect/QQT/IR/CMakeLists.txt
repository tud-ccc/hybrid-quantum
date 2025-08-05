add_mlir_dialect_library(QQTIR
        QQTBase.cpp
        QQTAttributes.cpp
        QQTTypes.cpp
        QQTOps.cpp
        
    DEPENDS
        QQTIncGen

    LINK_LIBS PUBLIC
        QILLRIR
        QuantumIR
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
