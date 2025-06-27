//===- Attributes.td - Quantum dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_ATTRIBUTES
#define QUANTUM_ATTRIBUTES

include "quantum-mlir/Dialect/Quantum/IR/QuantumBase.td"

//Need to replace the DeviceType parameter with this attribute in the future.
def Quantum_CouplingGraphAttr : Quantum_Attr<"CouplingGraph", "coupling_graph"> {
  let summary = "CouplingGraphAttr is used to represent the coupling graph of a quantum device.";
  let description = [{
    Examples:
    ```mlir
    #quantum.coupling_graph<5: [[0,1], [1,4]]>
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

#endif // QUANTUM_ATTRIBUTES
