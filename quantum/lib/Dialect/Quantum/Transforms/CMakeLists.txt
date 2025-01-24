add_mlir_dialect_library(QuantumTransforms
    Optimise.cpp
    Torch.cpp

DEPENDS
    QuantumIncGen
    QuantumTransformPassesIncGen

LINK_LIBS PUBLIC
    MLIRIR
    MLIRGPUDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRSPIRVTarget
    MLIRTransformUtils
)
