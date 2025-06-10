//===- Passes.td - QIR dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_PASSES
#define QIR_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeUGates : Pass<"qir-decompose-ugates", "ModuleOp"> {
  let summary = "Decompose the U gates in the `qir` dialect";

  let description = [{
  }];

  let constructor = "mlir::qir::createDecomposeUGatesPass()";
}

#endif // QIR_PASSES
