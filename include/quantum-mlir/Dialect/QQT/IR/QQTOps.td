//===- Ops.td - QQT dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the QQT dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QQT_OPS
#define QQT_OPS

include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.td"
include "quantum-mlir/Dialect/QQT/IR/QQTTypes.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QQT Op templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QQT dialect traits.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QQT operations.
//===----------------------------------------------------------------------===//

def QQT_PromoteOp : QQT_Op<
        "promote",
        [
          MemoryEffects<[
            MemRead,
            MemWrite
          ]>
        ]> {
  let summary = "Promote a qubit reference into a quantum register.";
  let description = [{
    Example:

    ```mlir
    %q0 = qqt.promote : !qqt.ref
    ```
  }];

  let results = (outs QQT_RefType:$result);
}

def QQT_DestructOp : QQT_Op<
        "destruct",
        [
          MemoryEffects<[
            MemRead,
            MemWrite
          ]>
        ]> {
  let summary = "Destruct a qubit reference into a quantum register.";
  let description = [{
    Example:

    ```mlir
    qqt.destruct(%ref) : (!qqt.ref) -> ()
    ```
  }];

  let arguments = (ins QQT_RefType:$input);
}

def QQT_LoadOp : QQT_Op<
        "load",
        [
          MemoryEffects<[
            MemRead
          ]>
        ]> {
  let summary = "Load a Quantum Qubit from a Qubit reference.";
  let description = [{
    Example:

    ```mlir
    %q0 = qqt.load %qref : !qqt.ref -> !quantum.qubit<1>
    ```
  }];
  let arguments = (ins QQT_RefType:$input);
  let results = (outs Quantum_QubitType:$result);
}

def QQT_StoreOp : QQT_Op<
        "store",
        [
          MemoryEffects<[
            MemWrite
          ]>
        ]> {
  let summary = "Store a Quantum Qubit into a Qubit reference.";
  let description = [{
    Example:

    ```mlir
    %q0 = qqt.store(%q, %qref) : !quantum.qubit<1>, !qqt.ref -> ()
    ```
  }];
  let arguments = (ins
    Quantum_QubitType:$qubit,
    QQT_RefType:$reference
  );
}

#endif // QQT_OPS
