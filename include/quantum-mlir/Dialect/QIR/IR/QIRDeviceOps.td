//===- QIRDeviceOps.td - QIR device-related operations --------------------*- tablegen -*-===//
//
// This file contains definitions for device-related operations in the QIR dialect.
//
//===--------------------------------------------------------------------//
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// QIR_DeviceOp: Defines a quantum device with a specific number of qubits and their coupling graph.
//===----------------------------------------------------------------------===//
def QIR_CircuitOp : QIR_Op<"circuit", [
  IsolatedFromAbove,
]> {
  let summary = "Quantum circuit operation with coupling graph";
  let description = [{ A quantum circuit operation that operates on a device with a specified coupling graph. }];
  let arguments = (ins QIR_DeviceType:$device);  // Only device is needed
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins "DeviceType":$device)>];

  // TODO: Add more builders if needed for different ways to create a circuit operation.
  let extraClassDeclaration = [{
    static CircuitOp create(Location location, DeviceType device) {
      // Logic to create the circuit operation
    }
  }];
}



def QIR_AllocateDeviceOp : QIR_Op<"allocate_device"> {
  let summary = "Allocate a quantum device with a specified number of qubits and coupling graph.";
  let description = [{
    Allocates a quantum device with a specified number of qubits and coupling graph.
  }];
  let arguments = (ins
    I64Attr:$num_qubits,
    ArrayAttr:$coupling_graph
  );
  let results = (outs QIR_DeviceType:$device);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "int64_t":$num_qubits, "ArrayAttr":$coupling_graph), [{
      build($_builder, $_state,
            DeviceType::get($_state.getContext(), num_qubits, coupling_graph),
            $_builder.getI64IntegerAttr(num_qubits),
            coupling_graph);
    }]>,
  ];
}
