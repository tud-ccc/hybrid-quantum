//===- Types.td - QQT dialect types -------------------------*- tablegen -*-===//
//
// This is the definitions file for the QQT dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QQT_TYPES
#define QQT_TYPES

include "quantum-mlir/Dialect/QQT/IR/QQTAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def QQT_RefType : QQT_Type<"QubitRef"> {
  let mnemonic = "ref";
  let summary = "Qubit register reference type";
}

#endif // QQT_TYPES
