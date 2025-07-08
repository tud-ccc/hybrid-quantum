//===- Types.td - QPU dialect types -------------------------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_TYPES
#define QPU_TYPES

include "quantum-mlir/Dialect/QPU/IR/QPUAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def QPU_DeviceType : QPU_Type<"Device"> {
  let mnemonic = "device";
  let description = "Quantum device type that holds information about the device size and coupling graph.";
  let parameters = (ins
    "int64_t":$qubits,
    "ArrayAttr":$edges
  );
  //let assemblyFormat = "`<` $qubits `,` $edges `>`";
  let extraClassDeclaration = [{
    bool hasStaticSize() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def QPU_CircuitType : QPU_Type<"Circuit"> {
  let mnemonic = "circuit";
  let description = "Quantum circuit type refers to a circuit that contains quantum operations.";
}

#endif // QPU_TYPES
