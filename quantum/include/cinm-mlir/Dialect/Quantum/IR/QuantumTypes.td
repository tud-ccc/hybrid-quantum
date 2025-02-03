//===- Types.td - Quantum dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Quantum_QubitType : Quantum_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "Multi-qubit register type";
  let parameters = (ins "unsigned":$size);
  let assemblyFormat = "`<` $size `>`";
}

def Quantum_MeasurementType : 1DTensorOf<[I1]>;

#endif
