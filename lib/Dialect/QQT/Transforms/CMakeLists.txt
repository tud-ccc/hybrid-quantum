add_mlir_dialect_library(QQTTransforms
        MoveLoadStore.cpp

    ENABLE_AGGREGATION

    DEPENDS
        QQTPassesIncGen

    LINK_LIBS PUBLIC
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
