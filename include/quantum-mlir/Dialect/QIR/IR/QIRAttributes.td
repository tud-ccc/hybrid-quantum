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


//Need to replace the DeviceType parameter with this attribute in the future.
def QIR_CouplingGraphAttr : QIR_Attr<"CouplingGraph", "coupling_graph", [
    OpAsmAttrInterface
  ]> {
  let summary = "CouplingGraphAttr is used to represent the coupling graph of a quantum device."
  let parameters = (ins "ArrayAttr":$edges);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Array":$edges), [{
      return $_get(edges.getContext(), value);
    }]>
  ];
  let assemblyFormat = "`<` $edges `>`";
}

#endif
