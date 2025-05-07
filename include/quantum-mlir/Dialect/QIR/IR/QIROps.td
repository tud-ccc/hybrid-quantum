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

include "quantum-mlir/Dialect/QIR/IR/QIRBase.td"
include "quantum-mlir/Dialect/QIR/IR/QIRTypes.td"


//===----------------------------------------------------------------------===//
// QIR device operations.
//===----------------------------------------------------------------------===//
def QIR_InitOp : QIR_Op<"init">{
  let summary = "Allocates a qubit.";
  let description = [{  }];
}

def QIR_SeedOp : QIR_Op<"seed">{
  let summary = "Set a seed for deterministic measurements.";
  let description = [{  }];
 let arguments = (ins I64:$seed);
}
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

def QIR_CNOTOp : Gate_Op<"CNOT"> {
  let summary = "CNOT gate operation";
  let description = [{
    The controlled-NOT (CNOT) gate flips the target qubit if the control qubit is in the |1> state.
  }];
  let arguments = (ins QIR_QubitType:$control, QIR_QubitType:$target);
}

def QIR_CZOp : Gate_Op<"Cz"> {
  let summary = "CZ gate operation";
  let description = [{
    The controlled-Z (CZ) gate
  }];
  let arguments = (ins QIR_QubitType:$control, QIR_QubitType:$target);
}

def QIR_RxOp : Gate_Op<"Rx"> {
  let summary = "Rx gate operation";
  let description = [{
    The X gate negates a qubit value.
  }];

  let arguments = (ins QIR_QubitType:$input, F64:$angle);
}

// Rz: a single-qubit rotation with an angle parameter.
def QIR_RzOp : Gate_Op<"Rz"> {
  let summary = "Rz gate operation";
  let description = [{
    The Rz gate rotates a qubit around the Z-axis by a specified angle.
  }];
  let arguments = (ins QIR_QubitType:$input, F64:$angle);
}

// Ry: a single-qubit rotation with an angle parameter.
def QIR_RyOp : Gate_Op<"Ry"> {
  let summary = "Ry gate operation";
  let description = [{
    The Ry gate rotates a qubit around the Y-axis by a specified angle.
  }];
  let arguments = (ins QIR_QubitType:$input, F64:$angle);
}

// U: a universal single-qubit gate with three angles.
def QIR_U3Op : Gate_Op<"U3"> {
  let summary = "U gate operation";
  let description = [{
    The U gate is a universal single-qubit gate that can represent any single-qubit operation.
  }];
  let arguments = (ins QIR_QubitType:$input, F64:$theta, F64:$phi, F64:$lambda);
}

def QIR_U2Op : Gate_Op<"U2"> {
  let summary = "U2 gate operation";
  let description = [{ A single-qubit gate with two parameters φ and λ. Equivalent to U3(π/2, φ, λ). }];
  let arguments = (ins QIR_QubitType:$input, F64:$phi, F64:$lambda);
}

def QIR_U1Op : Gate_Op<"U1"> {
  let summary = "U1 gate operation";
  let description = [{ A single-parameter unitary gate for phase rotation. Equivalent to Rz(λ). }];
  let arguments = (ins QIR_QubitType:$input, F64:$lambda);
}


def QIR_SOp : PrimitiveGate_Op<"S"> {
  let summary = "S gate operation";
  let description = [{ Phase gate (√Z). }];
}

def QIR_SdgOp : PrimitiveGate_Op<"Sdg"> {
  let summary = "S-dagger gate operation";
  let description = [{ Adjoint of the S gate. }];
}

def QIR_TOp : PrimitiveGate_Op<"T"> {
  let summary = "T gate operation";
  let description = [{ T gate (π/8 gate). }];
}

def QIR_TdgOp : PrimitiveGate_Op<"Tdg"> {
  let summary = "T-dagger gate operation";
  let description = [{ Adjoint of the T gate. }];
}

def QIR_CRzOp : Gate_Op<"CRz"> {
  let summary = "Controlled-Rz gate";
  let description = [{ Applies a Z rotation conditioned on the control qubit. }];
  let arguments = (ins QIR_QubitType:$control, QIR_QubitType:$target, F64:$angle);
}

def QIR_CRyOp : Gate_Op<"CRy"> {
  let summary = "Controlled-Ry gate";
  let description = [{ Applies a Y rotation conditioned on the control qubit. }];
  let arguments = (ins QIR_QubitType:$control, QIR_QubitType:$target, F64:$angle);
}

def QIR_CCXOp : Gate_Op<"CCX"> {
  let summary = "Toffoli (CCX) gate";
  let description = [{ Controlled-controlled-X gate. Flips target if both controls are |1>. }];
  let arguments = (ins QIR_QubitType:$control1, QIR_QubitType:$control2, QIR_QubitType:$target);
}

def QIR_BarrierOp : Gate_Op<"barrier"> {
  let summary = "Barrier operation";
  let description = [{ A barrier operation that prevents optimization across it. }];
  let arguments = (ins Variadic<QIR_QubitType>:$input);
}

// Swap: a two-qubit gate that exchanges the states of its inputs.
def QIR_SwapOp : Gate_Op<"swap"> {
  let summary = "Swap gate operation";
  let description = [{
    The swap gate exchanges the quantum states of two qubits.
  }];
  // inputs named control/target for nomenclature only. it means lhs, rhs in this case.
  let arguments = (ins
    QIR_QubitType:$control,
    QIR_QubitType:$target
  );
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
