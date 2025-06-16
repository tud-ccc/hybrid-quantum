//===- Attributes.td - QIR dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_ATTRIBUTES
#define QIR_ATTRIBUTES

include "quantum-mlir/Dialect/QIR/IR/QIRBase.td"

//Experimental: QIR_CouplingGraphAttr is used to represent the coupling graph of a quantum device.
//Need to replace the DeviceType parameter with this attribute in the future.
def QIR_CouplingGraphAttr : AttrDef<QIR_Dialect, "CouplingGraph", []> {
  let mnemonic = "coupling_graph";
  let parameters = (ins "ArrayAttr":$edges);
  let assemblyFormat = "`<` $edges `>`";
}

#endif
