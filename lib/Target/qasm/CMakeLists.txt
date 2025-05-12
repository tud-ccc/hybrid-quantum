add_mlir_translation_library(QASMToOpenQASM
    qir2qasm.cpp

    DEPENDS
    QIRIR

    LINK_LIBS PUBLIC
    MLIRIR
    QIRIR
    )

target_link_libraries(QASMToOpenQASM PUBLIC MLIRIR)
