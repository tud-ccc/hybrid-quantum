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
  let summary = "Allocates a qubit.";
  let description = [{  }];

  let results = (outs QIR_QubitType:$result);
}

def QIR_AllocResultOp : Memory_Op<"ralloc", [MemoryEffects<[MemRead]>]> {
  let summary = "Allocate n results into a result register.";
  let description = [{ Allocates a fixed number of results into a result register. }];

  let results = (outs QIR_ResultType:$result);
}

def QIR_ShowStateOp : Memory_Op<"show_state", [MemoryEffects<[MemRead]>]> {
  let summary = "Show the internal state of the simulator.";
  let description = [{  }];
}

//===----------------------------------------------------------------------===//
// QIR gate operations.
//===----------------------------------------------------------------------===//
class PrimitiveGate_Op<string mnemonic, list<Trait> traits = []>:Gate_Op<mnemonic, traits>{
  // Declare a mnemonic field so that derived definitions can override it.
  let arguments = (ins QIR_QubitType:$input);
}
        
def QIR_HOp : PrimitiveGate_Op<"H", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Hadamard gate operation";
  let description = [{
    The Hadamard gate creates an equal superposition of states.
  }];
}


def QIR_XOp : PrimitiveGate_Op<"X"> {
  let summary = "X gate operation";
  let description = [{
    The X gate negates a qubit value.
  }];
}

def QIR_ZOp : PrimitiveGate_Op<"Z"> {
  let summary = "Z gate operation";
  let description = [{
    The Z gate negates a qubit value.
  }];
}

def QIR_YOp : PrimitiveGate_Op<"Y"> {
  let summary = "Y gate operation";
  let description = [{
    The Y gate negates a qubit value.
  }];
}


def QIR_RXOp : Gate_Op<"RX"> {
  let summary = "RX gate operation";
  let description = [{
    The X gate negates a qubit value.
  }];

  let arguments = (ins QIR_QubitType:$input, F32:$theta);
}

// Rz: a single-qubit rotation with an angle parameter.
def QIR_RzOp : Gate_Op<"Rz"> {
  let summary = "Rz gate operation";
  let description = [{
    The Rz gate rotates a qubit around the Z-axis by a specified angle.
  }];
  let arguments = (ins QIR_QubitType:$input, F64:$angle);
}

// Swap: a two-qubit gate that exchanges the states of its inputs.
def QIR_SwapOp : Gate_Op<"Swap"> {
  let summary = "Swap gate operation";
  let description = [{
    The Swap gate exchanges the quantum states of two qubits.
  }];
  let arguments = (ins QIR_QubitType:$input1, QIR_QubitType:$input2);
}

def QIR_MeasureOp : QIR_Op<"measure", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Measure a single qubit.";
    let description = [{ }];

    let arguments = (ins
      QIR_QubitType:$input,
      QIR_ResultType:$result
    );
}

def QIR_ReadMeasurementOp : QIR_Op<"read_measurement", [MemoryEffects<[MemRead]>]> {
  let summary = "Read the measurement value from result memory";
  let description = [{ }];
  
  let arguments = (ins
    QIR_ResultType:$input
  );
  let results = (outs
    QIR_MeasurementType:$measurement
  );
}

def QIR_ResetOp : QIR_Op<"reset", [MemoryEffects<[MemRead]>]> {
  let summary = "Resets the qubit in the computational basis";
  let description = [{ }];
  
  let arguments = (ins QIR_QubitType:$input);
}

#endif
