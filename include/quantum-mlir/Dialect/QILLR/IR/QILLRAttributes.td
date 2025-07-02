//===- Attributes.td - QILLR dialect attributes ------------*- tablegen -*-===//
//
// This is the definitions file for the QILLR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QILLR_ATTRIBUTES
#define QILLR_ATTRIBUTES

include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.td"

class QILLR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QILLR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // QILLR_ATTRIBUTES
