add_mlir_translation_library(QIRToOpenQASM
    TargetQASMRegistration.cpp
    TargetQASM.cpp

    DEPENDS
    QIRIR

    LINK_LIBS PUBLIC
    MLIRIR
    QIRIR
    )

#target_link_libraries(QIRToOpenQASM PUBLIC MLIRIR)
