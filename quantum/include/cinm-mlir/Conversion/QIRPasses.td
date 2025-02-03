//===- Passes.td - QIR dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_CONVERSION_PASSES
#define QIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertQIRToLLVMPass : Pass<"convert-qir-to-llvm", "ModuleOp"> {
    let summary = "Perform a dialect conversion from QIR to LLVM MLIR";
    let constructor = "mlir::qir::createConvertQIRToLLVMPass()";
    let dependentDialects = [
        "mlir::LLVM::LLVMDialect",
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
    ];
}


#endif // QIR_CONVERSION_PASSES