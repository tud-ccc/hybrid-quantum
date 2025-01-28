//===- Passes.td - Quantum dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TRANSFORM_PASSES
#define QUANTUM_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def QuantumOptimisePass : Pass<"quantum-optimise", ""> {
  let summary = "Optimise the `quantum` dialect";
  let description = [{
  }];
}

def QuantumTorchPass : Pass<"quantum-torch", "">{
  let summary = "Convert torch dialect to quantum dialect";
  let description = [{
  }];
}

#endif // QUANTUM_TRANSFORM_PASSES
