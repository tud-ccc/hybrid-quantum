//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_PASSES
#define QUANTUM_PASSES

include "mlir/Pass/PassBase.td"

def QuantumOptimise : Pass<"quantum-optimise", "ModuleOp"> {
  let summary = "Optimise the `quantum` dialect";

  let description = [{
  }];

  let constructor = "mlir::quantum::createQuantumOptimisePass()";
}

def MultiQubitLegalization : Pass<"quantum-multi-qubit-legalize", "ModuleOp"> {
  let summary = "Legalize multi-qubit registers in the `quantum` dialect";

  let description = [{
  This pass legalizes multi-qubit quantum registers in the quantum dialect
  such that they can be amenable to further lowerings.
  }];

  let constructor = "mlir::quantum::createMultiQubitLegalizationPass()";

  let dependentDialects = [
    "tensor::TensorDialect"
  ];
}

def ScfToRVSDG : Pass<"scf-to-rvsdg", "ModuleOp"> {
  let summary = "Transform `scf.if` and `scf.for` to RVSDG variants of the `quantum` dialect.";

  let description = [{
  This pass transform structured control flow to the regionalized value state dependence
  graph (RVSDG) elements from the `quantum` dialect.
  }];

  let constructor = "mlir::quantum::createScfToRVSDGPass()";

  let dependentDialects = [
    "scf::SCFDialect"
  ];
}

#endif // QUANTUM_PASSES
