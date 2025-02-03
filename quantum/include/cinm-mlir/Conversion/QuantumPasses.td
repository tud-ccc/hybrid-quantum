//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANUTM_CONVERSION_PASSES
#define QUANTUM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertQuantumToLLVMPass : Pass<"convert-quantum-to-llvm", "ModuleOp"> {
    let summary = "Perform a dialect conversion from quantum to LLVM (QIR)";
    let constructor = "mlir::quantum::createConvertQuantumToLLVMPass()";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
    ];
}

def ConvertQuantumToQIRPass : Pass<"convert-quantum-to-qir", "ModuleOp"> {
    let summary = "Perform a dialect conversion from Quantum to QIR";
    let constructor = "mlir::quantum::createConvertQuantumToQIRPass()";
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
    ];
}


#endif // Quantum_CONVERSION_PASSES//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//