//===- Types.td - Quantum dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "QuantumBase.td"

def Quantum_AsyncToken : Quantum_Type<"AsyncToken"> {
  let mnemonic = "async.token";
  let summary = "";
}

//UNUSED SO FAR. In the future, we can define a register of qubit on operations. 
def Quantum_Qubit : Quantum_Type<"qubit"> {
  let mnemonic = "qubit";
  let summary = "Multi-qubit register type";
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

def Quantum_ObservableType : Quantum_Type<"observable"> {
    let mnemonic = "obs";
    let summary = "A quantum observable for use in measurements.";
}

def Quantum_ResultType : Quantum_Type<"result"> {
    let mnemonic = "result";
    let summary = "A quantum measurement result.";
}

def Quantum_MeasurementType : 1DTensorOf<[I1]>;

#endif
