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

def Quantum_Qubit : Quantum_Type<"qubit"> {
  let mnemonic = "qubit";
  let summary = "Single Quantum qubit. Some opearations only work on 1 qubits";
}

//UNUSED SO FAR. In the future, we can define a register of qubit on operations. 
def Quantum_QubitArray : Quantum_Type<"nqubit"> {
  let mnemonic = "nqubit";
  let summary = "Multi-qubit register type";
}

def Quantum_ObservableType : Quantum_Type<"observable"> {
    let mnemonic = "obs";
    let summary = "A quantum observable for use in measurements.";
}

def Quantum_CircuitType : Quantum_Type<"circuitTyp"> {
    let summary = "Quantum circuit representation";
    let mnemonic = "circ";
}

def Quantum_StateType : Quantum_Type<"state"> {
    let summary = "Quantum circuit representation";
    let mnemonic = "state";
}

def Quantum_ResultType : Quantum_Type<"result"> {
    let mnemonic = "result";
    let summary = "A quantum measurement result.";
}

#endif
