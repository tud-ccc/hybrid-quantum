//===- Types.td - QIR dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_TYPES
#define QIR_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "QIRBase.td"

def QIR_QubitType : QIR_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "qubit type";
}

def QIR_ResultType : QIR_Type<"Result"> {
    let mnemonic = "result";
    let summary = "A quantum measurement result.";
}

#endif
