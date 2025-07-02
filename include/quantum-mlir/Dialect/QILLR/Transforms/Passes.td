//===- Passes.td - QILLR dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the QILLR dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QILLR_PASSES
#define QILLR_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeUGates : Pass<"qillr-decompose-ugates", "ModuleOp"> {
  let summary = "Decompose the U gates in the `qillr` dialect";

  let description = [{
  }];

  let constructor = "mlir::qillr::createDecomposeUGatesPass()";
}

#endif // QILLR_PASSES
