//===- Passes.td - QIR dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_PASSES
#define QIR_PASSES

include "mlir/Pass/PassBase.td"

def QIROptimise : Pass<"qir-optimise", "ModuleOp"> {
  let summary = "Optimise the `qir` dialect";

  let description = [{
  }];

  let constructor = "mlir::qir::createQIROptimisePass()";
}

#endif // QIR_PASSES
