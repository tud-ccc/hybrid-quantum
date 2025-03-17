//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the cinm-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_CONVERSION_PASSES
#define QUANTUM_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertQuantumToQIR : Pass<"convert-quantum-to-qir"> {
    let summary = "Perform a dialect conversion from Quantum to QIR";

    let constructor = "mlir::createConvertQuantumToQIRPass()";

    let dependentDialects = [
        "quantum::QuantumDialect",
        "qir::QIRDialect",
        "tensor::TensorDialect"
    ];
}

def ConvertQIRToLLVM : Pass<"convert-qir-to-llvm"> {
    let summary = "Perform a dialect conversion from QIR to LLVM MLIR";

    let constructor = "mlir::createConvertQIRToLLVMPass()";
    
    let dependentDialects = [
        "qir::QIRDialect",
        "LLVM::LLVMDialect",
        "tensor::TensorDialect"
    ];
}


#endif // QUANTUM_MLIR_CONVERSION_PASSES