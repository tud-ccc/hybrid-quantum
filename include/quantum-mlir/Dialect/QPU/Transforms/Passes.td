//===- Passes.td - QPU dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_PASSES
#define QPU_PASSES

include "mlir/Pass/PassBase.td"

def SabreSwap : Pass<"sabre-swap", "QPUModuleOp"> {
    let summary = "Implements the SABRE swap algorithm";

    let description = [{
    }];

    let constructor = "mlir::qpu::createSabreSwapPass()";

    let dependentDialects = [
        "quantum::QuantumDialect"
    ];
}

#endif // QPU_PASSES
