//===- Attributes.td - QIR dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_ATTRIBUTES
#define QIR_ATTRIBUTES

include "quantum-mlir/Dialect/QIR/IR/QIRBase.td"

class QIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // QIR_ATTRIBUTES
