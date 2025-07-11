add_mlir_dialect_library(RVSDGIR
        RVSDGAttributes.cpp
        RVSDGBase.cpp
        RVSDGOps.cpp
        RVSDGTypes.cpp
        
    DEPENDS
        RVSDGIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRArithDialect
        MLIRArithTransforms
)
