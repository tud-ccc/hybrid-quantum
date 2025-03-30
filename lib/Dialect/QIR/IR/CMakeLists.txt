add_mlir_dialect_library(QIRIR
        QIRAttributes.cpp
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
