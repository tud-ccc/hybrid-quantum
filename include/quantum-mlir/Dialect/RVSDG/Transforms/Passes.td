//===- Passes.td - RVSDG dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the RVSDG dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef RVSDG_PASSES
#define RVSDG_PASSES

include "mlir/Pass/PassBase.td"

def ControlFlowHoisting : Pass<"control-flow-hoisting", "ModuleOp"> {
  let summary = "If an operation exists on all regions of a `rvsdg.gamma` it is hoisted in front of the conditional.";

  let description = [{
  This pass hoists operations that are idendical and in all regions of a `rvsdg.gamma` out of the regions.
  The capturing arguments will be rewritten if necessary.
  }];

  let constructor = "mlir::rvsdg::createControlFlowHoistingPass()";
}

#endif // RVSDG_PASSES
