//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the quantum-mlir dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_MLIR_CONVERSION_PASSES
#define QUANTUM_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertQuantumToQILLR : Pass<"convert-quantum-to-qillr"> {
    let summary = "Perform a dialect conversion from Quantum to QILLR";

    let constructor = "mlir::createConvertQuantumToQILLRPass()";

    let dependentDialects = [
        "quantum::QuantumDialect",
        "qillr::QILLRDialect",
        "tensor::TensorDialect"
    ];
}

def ConvertQILLRToLLVM : Pass<"convert-qillr-to-llvm"> {
    let summary = "Perform a dialect conversion from QILLR to LLVM MLIR";

    let constructor = "mlir::createConvertQILLRToLLVMPass()";
    
    let dependentDialects = [
        "qillr::QILLRDialect",
        "LLVM::LLVMDialect",
        "tensor::TensorDialect"
    ];
}

def ConvertQILLRToQuantum : Pass<"lift-qillr-to-quantum"> {
    let summary = "Perform a dialect lifting from QILLR to Quantum";

    let constructor = "mlir::createConvertQILLRToQuantumPass()";

    let dependentDialects = [
        "quantum::QuantumDialect",
        "qillr::QILLRDialect",
        "mlir::tensor::TensorDialect",
        "mlir::scf::SCFDialect"
    ];
}

def ConvertScfToRVSDG : Pass<"convert-scf-to-rvsdg"> {
  let summary = "Transform `scf.if` and `scf.for` to RVSDG variants of the rvsdg dialect.";

  let description = [{
    This pass transforms structured control flow to the regionalized value state dependence
    graph (RVSDG) dialect.
  }];

  let constructor = "mlir::createConvertScfToRVSDGPass()";

  let dependentDialects = [
    "scf::SCFDialect",
    "rvsdg::RVSDGDialect"
  ];
}

#endif // QUANTUM_MLIR_CONVERSION_PASSES
