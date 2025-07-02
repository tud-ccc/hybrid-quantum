//===- Attributes.td - QPU dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_ATTRIBUTES
#define QPU_ATTRIBUTES

include "quantum-mlir/Dialect/QPU/IR/QPUBase.td"

//Need to replace the DeviceType parameter with this attribute in the future.
def QPU_CouplingGraphAttr : QPU_Attr<"CouplingGraph", "coupling_graph"> {
  let summary = "CouplingGraphAttr is used to represent the coupling graph of a QPU device.";
  let description = [{
    Examples:
    ```mlir
    #qpu.coupling_graph<5: [[0,1], [1,4]]>
    ```
  }]; 
  let parameters = (ins
    "IntegerAttr":$qubits,
    "ArrayAttr":$edges
  );
  let assemblyFormat = "`<` $qubits `,` $edges `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "IntegerAttr":$qubits,
                                        "ArrayAttr":$edges), [{
      return $_get(qubits.getContext(), qubits, edges);
    }]>
  ];
  let skipDefaultBuilders = 0;
}

#endif // QPU_ATTRIBUTES
