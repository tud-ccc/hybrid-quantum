add_mlir_dialect_library(QIRIR
        QIRBase.cpp
        QIROps.cpp
        QIRTypes.cpp

    DEPENDS
        QIRIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
