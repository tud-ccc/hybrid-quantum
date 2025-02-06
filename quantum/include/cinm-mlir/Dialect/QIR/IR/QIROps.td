//===- Ops.td - QIR dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the QIR dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QIR_OPS
#define QIR_OPS

include "mlir/IR/EnumAttr.td"
//include "mlir/IR/OpAsmInterface.td"
//include "mlir/IR/SymbolInterfaces.td"
//include "mlir/Interfaces/CallInterfaces.td"
//include "mlir/Interfaces/ControlFlowInterfaces.td"
//include "mlir/Interfaces/FunctionInterfaces.td"
//include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "cinm-mlir/Dialect/QIR/IR/QIRBase.td"
include "cinm-mlir/Dialect/QIR/IR/QIRTypes.td"

//===----------------------------------------------------------------------===//
// QIR memory operations.
//===----------------------------------------------------------------------===//
def QIR_AllocOp : Memory_Op<"alloc", [MemoryEffects<[MemRead]>]> {
  let summary = "Allocate n qubits into a qubit register.";
  let description = [{ Allocates a fixed number of qubits into a qubit register. }];

  let results = (outs QIR_QubitType:$result);
}

def QIR_AllocResultOp : Memory_Op<"ralloc", [MemoryEffects<[MemRead]>]> {
  let summary = "Allocate n results into a result register.";
  let description = [{ Allocates a fixed number of results into a result register. }];

  let results = (outs QIR_ResultType:$result);
}

//===----------------------------------------------------------------------===//
// QIR gate operations.
//===----------------------------------------------------------------------===//
class PrimitiveGate_Op<string mnemonic, list<Trait> traits = []> :
        Gate_Op<mnemonic, traits>;
        
def QIR_HOp : PrimitiveGate_Op<"H", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Hadamard gate operation";
  let description = [{
    The Hadamard gate creates an equal superposition of states.
  }];

  let arguments = (ins QIR_QubitType:$input);
}

def QIR_MeasureOp : QIR_Op<"measure", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Measure a single qubit.";
    let description = [{ }];

    let arguments = (ins QIR_QubitType:$input, QIR_ResultType:$result);
}

def QIR_ReadMeasurementOp : QIR_Op<"read_measurement", [MemoryEffects<[MemRead]>]> {
  let summary = "Read the measurement value from result memory";
  let description = [{ }];
  
  let arguments = (ins QIR_ResultType:$input);
  let results = (outs I1:$measurement);
}

#endif
