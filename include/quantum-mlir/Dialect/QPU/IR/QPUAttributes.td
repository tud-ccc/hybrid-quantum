//===- Attributes.td - QPU dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_ATTRIBUTES
#define QPU_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "quantum-mlir/Dialect/QPU/IR/QPUBase.td"

//===----------------------------------------------------------------------===//
// QPU Target Attributes.
//===----------------------------------------------------------------------===//

def QPU_TargetAttr : QPU_Attr<"Target", "target"> {
  let summary = "Generic QPU target attribute with coupling graph and qubits.";
  let description = [{

  }];
  let parameters = (ins
    "IntegerAttr":$qubits,
    "ArrayAttr":$edges
  );
  let assemblyFormat = [{
    `<`
    `qubits` `=` $qubits `,`
    `coupling_graph` `=` $edges
    `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "IntegerAttr":$qubits,
                                        "ArrayAttr":$edges), [{
      return $_get(qubits.getContext(), qubits, edges);
    }]>
  ];
  let skipDefaultBuilders = 0;
}

def QPU_TargetArrayAttr : TypedArrayAttrBase<QPU_TargetAttr,
  "array of QPU target attributes">;

def QPU_NonEmptyTargetArrayAttr :
  ConfinedAttr<QPU_TargetArrayAttr, [ArrayMinCount<1>]>;

#endif // QPU_ATTRIBUTES
