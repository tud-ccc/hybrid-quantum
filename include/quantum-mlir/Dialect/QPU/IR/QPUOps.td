//===- Ops.td - QPU dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_OPS
#define QPU_OPS

include "quantum-mlir/Dialect/QPU/IR/QPUBase.td"
include "quantum-mlir/Dialect/QPU/IR/QPUTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QPU OP templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPU dialect traits.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPU gate operations.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPU_DeviceOp: Defines a quantum device with a specific number of qubits
// and their coupling graph.
//===----------------------------------------------------------------------===//

def QPU_CircuitOp : QPU_Op<"circuit", [
    IsolatedFromAbove
  ]> {
  let summary = "Quantum circuit operation with coupling graph";
  let description = [{ A quantum circuit operation that operates on a device with a specified coupling graph. }];
  let arguments = (ins
    SymbolNameAttr:$name,
    OptionalAttr<TypeAttrOf<QPU_DeviceType>>:$device_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    TypeAttrOf<FunctionType>:$circuit_type
  );
  let results = (outs
    QPU_CircuitType:$circuit
  );
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins
    "StringRef":$name,
    "DeviceType":$device,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    static CircuitOp create(Location location, StringRef name, DeviceType device,
                         FunctionType type, ArrayRef<NamedAttribute> attrs = {});

    //===------------------------------------------------------------------===//
    // Methods to query the circuit function
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getCircuitType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getCircuitType().getResults(); }
  }];
}

def QPU_DeviceOp : QPU_Op<"device"> {
  let summary = "Allocate a quantum device with a specified number of qubits and coupling graph.";
  let description = [{
    Allocates a quantum device with a specified number of qubits and coupling graph.
  }];
  let arguments = (ins
    QPU_CouplingGraphAttr:$coupling_graph
  );
  let results = (outs QPU_DeviceType:$device);
  let assemblyFormat = [{
    `(` `)` attr-dict `:` `->` type($device)
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "CouplingGraphAttr":$coupling_graph)>
  ];
}

def QPU_InstantiateOp : QPU_Op<"instantiate"> {
  let summary = "Reifies a quantum circuit with a specific device.";
  let description = [{
    Reifies a quantum circuit with a specific device. The result is a
    circuit that uses physical qubits and a circuit that obeys
    the device's coupling topology.
  }];
  let arguments = (ins
    QPU_DeviceType:$device,
    QPU_CircuitType:$circuit
  );
  let results = (outs
    QPU_CircuitType:$result
  );
}

def QPU_ExecuteOp : QPU_Op<"execute"> {
  let summary = "Executed a quantum circuit.";
  let description = [{
    Execute a quantum circuit. If the circuit has free parameters
    these must be provided to the execute operation.
  }];
  let arguments = (ins
    QPU_CircuitType:$circuit,
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>);
}

#endif // QPU_OPS
