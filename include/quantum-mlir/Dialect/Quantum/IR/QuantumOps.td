//===- Ops.td - Quantum dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "quantum-mlir/Dialect/Quantum/IR/QuantumTypes.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Quantum OP templates
//===----------------------------------------------------------------------===//

class Gate_Op<string mnemonic, list<Trait> traits = []>
        : Quantum_Op<mnemonic, traits>;

class Memory_Op<string mnemonic, list<Trait> traits = []>
        : Quantum_Op<mnemonic, traits # [ConditionallySpeculatable]>
{
    code extraClassDeclaration = [{
        ::mlir::Speculation::Speculatability getSpeculatability() const {
            return ::mlir::Speculation::NotSpeculatable;
        }
    }];
}

class Observable_Op<string mnemonic, list<Trait> traits = []>
        : Quantum_Op<mnemonic, traits # [Pure]>;

class Measurement_Op<string mnemonic, list<Trait> traits = []>
        : Quantum_Op<mnemonic, traits>; 

//===----------------------------------------------------------------------===//
// Quantum dialect traits.
//===----------------------------------------------------------------------===//

// Inject no clone theorem verifier into operations
// that produce qubits
def NoClone : NativeOpTrait<"NoClone"> {
    let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Quantum memory operations.
//===----------------------------------------------------------------------===//

def Quantum_AllocOp : Memory_Op<
        "alloc",
        [
          MemoryEffects<[
            MemRead
          ]>,
          NoClone]> {
  let summary = "Allocate n qubits into a quantum register.";
  let description = [{
    Allocates a fixed number of N qubits into a single quantum register.

    Example:

    ```mlir
    %q0 = quantum.alloc : !quantum.qubit<5>
    ```
  }];

  let results = (outs Quantum_QubitType:$result);
}

def Quantum_DeallocateOp : Memory_Op<
        "deallocate",
        [
          MemoryEffects<[
            MemRead,
            MemWrite
          ]>]> {
    let summary = "Deallocate a quantum register.";
    let description = [{
    Use this operation on all qubit registers that are not measured.
    Deallocated qubits no longer can be used.

    Example:
    
    ```mlir
    %q0 = quantum.alloc : !quantum.qubit<5> 
    %quantum.deallocate (%q0) : !quantum.qubit<5> -> ()
    ```
    }];

    let arguments = (ins Quantum_QubitType:$input);
}

def Quantum_SplitOp : Memory_Op<
        "split",
        [
          MemoryEffects<[
            MemRead,
            MemWrite
          ]>,
          NoClone]> {
    let summary = "Split a qubit register into two sub-registers.";
    let description = [{
    Example: 

    ```mlir
    %q = quantum.alloc : !quantum.qubit<5>
    %q0, %q1 = quantum.split (%q) : (!quantum.qubit<5>) -> (!quantum.qubit<2>, !quantum.qubit<3>)
    ```
    }];

    let arguments = (ins Quantum_QubitType:$input);
    let results = (outs
      Variadic<Quantum_QubitType>:$result
    );
    let hasVerifier = 0;
}

def Quantum_MergeOp : Memory_Op<
        "merge",
        [
          MemoryEffects<[
            MemRead,
            MemWrite
          ]>,
          NoClone]> {
    let summary = "Combine qubit registers into a single one.";
    let description = [{
    Example:

    ```mlir
    %lhs = "quantum.alloc" () : () -> !quantum.qubit<3>
    %rhs = "quantum.alloc" () : () -> !quantum.qubit<2>
    %q = "quantum.merge" (%lhs, %rhs) : (!quantum.qubit<3>, !quantum.qubit<2>) -> (!quantum.qubit<5>)
    ```
    }];

    let arguments = (ins
        Quantum_QubitType:$lhs,
        Quantum_QubitType:$rhs
    );

    let results = (outs Quantum_QubitType:$result);

    let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Quantum gate operations.
//===----------------------------------------------------------------------===//

class PrimitiveGate_Op<string mnemonic, list<Trait> traits = []> :
        Gate_Op<mnemonic, traits # [NoClone]>;
        
def Quantum_HOp : PrimitiveGate_Op<
        "H",
        [
          NoMemoryEffect,
          AllTypesMatch<["input", "result"]>]> {
  let summary = "Hadamard gate operation";
  let description = [{
    Example:
    %q1 = "quantum.H" (%q0): !quantum.qubit<1> -> !quantum.qubit<1>
  }];

  let arguments = (ins Quantum_QubitType:$input);
  let results = (outs Quantum_QubitType:$result);
  let hasFolder = 1;
}

def Quantum_CNOTOp : PrimitiveGate_Op<
        "CNOT",
        [
          NoMemoryEffect]> {
  let summary = "CNOT gate operation";
  let description = [{
    Example:
    ```mlir
    %qa1, %qb1 = "quantum.CNOT" (%qa0, %qb0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<1>, !quantum.qubit<1>
    ```
  }];

  let arguments = (ins
    Quantum_QubitType:$input,
    Quantum_QubitType:$target
  );
  let results = (outs
    Quantum_QubitType:$control_out,
    Quantum_QubitType:$target_out
  );
}

def Quantum_XOp : PrimitiveGate_Op<
        "X",
        [
          NoMemoryEffect,
          AllTypesMatch<["input", "result"]>]> {
  let summary = "X gate operation (NOT gate)";
  let description = [{%out = "quantum.X"(%qubit) : !quantum.qubit<1> -> !quantum.qubit<1>}]; 
  let arguments = (ins Quantum_QubitType:$input);
  let results = (outs Quantum_QubitType:$result);
  let hasFolder = 1;
}

def Quantum_YOp : PrimitiveGate_Op<
        "Y",
        [
          NoMemoryEffect,
          AllTypesMatch<["input", "result"]>]> {
  let summary = "Y gate operation (Pauli-Y gate)";
  let description = [{%out = "quantum.Y"(%qubit) : !quantum.qubit<1> -> !quantum.qubit<1>}];
  let arguments = (ins Quantum_QubitType:$input);
  let results = (outs Quantum_QubitType:$result);
}

def Quantum_ZOp : PrimitiveGate_Op<
        "Z",
        [
          NoMemoryEffect,
          AllTypesMatch<["input", "result"]>]> {
  let summary = "Z gate operation (Pauli-Z gate)";
  let description = [{%out = "quantum.Z"(%qubit) : !quantum.qubit<1> -> !quantum.qubit<1>}];
  let arguments = (ins Quantum_QubitType:$input);
  let results = (outs Quantum_QubitType:$result);
}

def Quantum_SWAPOp : PrimitiveGate_Op<
        "SWAP",
        [
          AllTypesMatch<["lhs", "rhs", "result1", "result2"]>]> {
  let summary = "SWAP gate";
  let description = [{
    Swaps the states of two qubits.
  }];
  let arguments = (ins
    Quantum_QubitType:$lhs,
    Quantum_QubitType:$rhs
  );
  let results = (outs
    Quantum_QubitType:$result1,
    Quantum_QubitType:$result2
  );
}

def Quantum_MeasureSingleOp : Memory_Op<
        "measure_single",
        [
          MemoryEffects<[
            MemRead
          ]>,
          AllTypesMatch<["input", "result"]>,
          NoClone]> {
    let summary = "Measure a single qubit.";
    let description = [{ 
      Return the measurement outcome of a single quibit in the basis dimension.
      The qubit can no longer be used.

      Example:
      
      ```mlir
      %q0 = quantum.allocate : !quantum.qubit<1>
      %m, %qm = quantum.measure_single (%q0) : !quantum.qubit<1> -> (i1, !quantum.qubit<1>)
    }];

    let arguments = (ins Quantum_SingleQubitType:$input);
    let results = (outs
      I1:$measurement,
      Quantum_SingleQubitType:$result
    );
}

def Quantum_MeasureOp : Memory_Op<
        "measure",
        [
          MemoryEffects<[
            MemRead
          ]>,
          AllTypesMatch<["input", "result"]>,
          NoClone]> {
    let summary = "Measure a qubit register of size N.";
    let description = [{
      %q0 = quantum.allocate : !quantum.qubit<2>
      %m, %qm = quantum.measure_single (%q0) : !quantum.qubit<2> -> (tensor<2xi1>, !quantum.qubit<2>)
    }];

    let arguments = (ins Quantum_QubitType:$input);
    let results = (outs
      Quantum_MeasurementType:$measurement,
      Quantum_QubitType:$result
    );
}

#endif // QUANTUM_OPS
