//===- Passes.td - QIR dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_TRANSFORM_PASSES
#define QIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def QIROptimisePass : Pass<"qir-optimise", ""> {
  let summary = "Optimise the `QIR` dialect";
  let description = [{
  }];
}

#endif // QIR_TRANSFORM_PASSES
