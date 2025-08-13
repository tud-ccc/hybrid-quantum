//===- Passes.td - QQT dialect passes -----------------------*- tablegen -*-===//
//
// This is the definitions file for the QQT dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QQT_PASSES
#define QQT_PASSES

include "mlir/Pass/PassBase.td"

def LoadStoreMove : Pass<"hoist-load-store", "ModuleOp"> {
  let summary = "Move load and store operations of the qqt dialect";

  let description = [{
  }];

  let constructor = "mlir::qqt::createLoadStoreMovePass()";
}

def LoadStoreElimination : Pass<"eliminate-load-store", "ModuleOp"> {
  let summary = "Eliminate loads immediately following a store operation of the qqt dialect";

  let description = [{
  }];

  let constructor = "mlir::qqt::createLoadStoreEliminationPass()";
}

#endif // QQT_PASSES
