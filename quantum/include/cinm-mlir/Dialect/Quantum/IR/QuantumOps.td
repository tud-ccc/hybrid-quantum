//===- Ops.td - Quantum dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.td"
include "cinm-mlir/Dialect/Quantum/IR/QuantumTypes.td"
include "cinm-mlir/Dialect/Quantum/IR/QuantumAttributes.td"

include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def NamedObservable : I32EnumAttr<"NamedObservable",
    "Known named observables",
    [
        I32EnumAttrCase<"Identity", 0>,
        I32EnumAttrCase<"PauliX",   1>,
        I32EnumAttrCase<"PauliY",   2>,
        I32EnumAttrCase<"PauliZ",   3>,
        I32EnumAttrCase<"Hadamard", 4>,
    ]> {
    let cppNamespace = "::mlir::quantum";
    let genSpecializedAttr = 0;
}

def NamedObservableAttr : EnumAttr<Quantum_Dialect, NamedObservable, "named_observable">;


//===----------------------------------------------------------------------===//
// Quantum memory operations.
//===----------------------------------------------------------------------===//
def Quantum_AllocOp : Quantum_Op<"alloc", [NoMemoryEffect]> {
  let summary = "Allocate n qubits into a quantum register.";
  let description = [{
    Allocates a fixed number of qubits into a quantum register.
  }];

  let results = (outs Quantum_Qubit:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def Quantum_DeallocateOp : Memory_Op<"deallocate"> {
    let summary = "Deallocate a quantum register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_Qubit:$qreg
    );

    let assemblyFormat = [{
        $qreg attr-dict `:` type(operands)
    }];
}

def Quantum_ExtractOp : Quantum_Op<"extract", [NoMemoryEffect]> {
    let summary = "Extract a qubit value or a range of qubits from a register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_Qubit:$qreg,
        Variadic<Index>:$idx
    );

    let results = (outs
        Quantum_Qubit:$out
    );

    let assemblyFormat = [{
        $qreg (`[` $idx^ `]`)? attr-dict `:` type($qreg) `->` type($out) }];

    let hasVerifier = 1;
}

def Quantum_InsertOp : Memory_Op<"insert", [NoMemoryEffect]> {
    let summary = "Update the qubit value of a register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_Qubit:$qreg,
        Optional<I32>:$idx,
        OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$idx_attr,
        Quantum_Qubit:$qubit
    );

    let results = (outs
        Quantum_Qubit:$out
    );

    let assemblyFormat = [{
        $qreg `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` type($qreg) `,` type($qubit) `->` type(results)
    }];

    let hasVerifier = 1;
}

def Quantum_CastOp : Memory_Op<"cast", [Pure]> {
  let summary = "tensor cast operation";
  let description = [{
    Example:

    ```mlir
    // Convert from unknown rank to rank 2 with unknown dimension sizes.
    %2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>
  }];

  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def Quantum_DimOp : Memory_Op<"dim", [NoMemoryEffect]> {
  let summary = "dimension index operation";
  let description = [{
    The `tensor.dim` operation takes a tensor and a dimension operand of type
    `index`. It returns the size of the requested dimension of the given
    tensor. If the dimension index is out of bounds, the behavior is undefined.

    The specified tensor type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = arith.constant 0 : index
    %x = tensor.dim %A, %c0 : tensor<4x?xf32>

    // Return the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = tensor.dim %A, %c1 : memref<4x?xf32>

    // Equivalent generic form:
    %x = "tensor.dim"(%A, %c0) : (memref<4x?xf32>, index) -> index
    %y = "tensor.dim"(%A, %c1) : (memref<4x?xf32>, index) -> index
    ```
  }];

  let arguments = (ins Quantum_Qubit:$source,
                       I32:$index);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    attr-dict $source `,` $index `:` type($source)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>
  ];
}

//===----------------------------------------------------------------------===//
// Quantum gate operations.
//===----------------------------------------------------------------------===//
class PrimitiveGate_Op<string mnemonic, list<Trait> traits = []> :
        Gate_Op<mnemonic, traits # [Unitary]>;
        
def Quantum_HOp : PrimitiveGate_Op<"H", [Pure, SameOperandsAndResultType, Hermitian]> {
  let summary = "Hadamard gate operation";
  let description = [{
    The Hadamard gate creates an equal superposition of states.
    It is a single-qubit unitary operation that is also Hermitian (self-inverse).
    Its 2x2 matrix representation in the computational basis is:
        [[1,  1],
         [1, -1]] / sqrt(2)

    This operation is applied to a single qubit.

    Example:
    ```mlir
    %q1 = quantum.H %q0
    ```
  }];

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}


def Quantum_CNOTOp : PrimitiveGate_Op<"CNOT", [Pure]> {
  let summary = "Quantum CNOT gate";
  let description = [{
    Takes two 1-qubit inputs, and applies the CNOT gate,
    with the first qubit as control, and second as target.

    Example:
    ```mlir
    %qa1, %qb1 = quantum.CNOT %qa0, %qb0
    ```
  }];

  let arguments = (ins
    Quantum_Qubit:$input,
    Quantum_Qubit:$target
  );
  let results = (outs
    Quantum_Qubit:$control_out,
    Quantum_Qubit:$target_out
  );

  let assemblyFormat = [{ $input `,` $target attr-dict `:`  type($input) `,` type($target)  `->` type($control_out) `,` type($target_out)}];
  let hasVerifier = 1;
}

def Quantum_XOp : PrimitiveGate_Op<"X", [Pure, SameOperandsAndResultType, Hermitian]> {
  let summary = "X gate operation (NOT gate)";
  let description = [{
    The X gate (also known as NOT gate or Pauli-X gate) flips the state of a qubit.
    It can be applied to a single qubit or a multi-qubit register.
    When applied to a register, it acts on all qubits in the register.

    The matrix representation of the X gate is:
    [[0, 1],
     [1, 0]]

    Example:
    ```mlir
    // Apply X to a single qubit
    %1 = quantum.X %qubit : !quantum.qubit

    // Apply X to each qubit in a 3-qubit register
    %2 = quantum.X %register : !quantum.nqubit<3>
    ```
  }];

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];

  let hasVerifier = 1;
}

def Quantum_YOp : PrimitiveGate_Op<"Y", [Pure, SameOperandsAndResultType, Hermitian]> {
  let summary = "Y gate operation (Pauli-Y gate)";
  let description = [{
    The Y gate (also known as Pauli-Y gate) applies a phase shift of π/2 around the Y-axis of the Bloch sphere.
    It can be applied to a single qubit or a multi-qubit register.
    When applied to a register, it acts on all qubits in the register.

    The matrix representation of the Y gate is:
    [[0, -i],
     [i, 0]]

    Example:
    ```mlir
    // Apply Y to a single qubit
    %1 = quantum.Y %qubit : !quantum.qubit

    // Apply Y to each qubit in a 3-qubit register
    %2 = quantum.Y %register : !quantum.nqubit<3>
    ```
  }];

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def Quantum_ZOp : PrimitiveGate_Op<"Z", [Pure, SameOperandsAndResultType, Hermitian]> {
  let summary = "Z gate operation (Pauli-Z gate)";
  let description = [{
    The Z gate (also known as Pauli-Z gate) applies a phase shift of π around the Z-axis of the Bloch sphere.
    It can be applied to a single qubit or a multi-qubit register.
    When applied to a register, it acts on all qubits in the register.

    The matrix representation of the Z gate is:
    [[1, 0],
     [0, -1]]

    Example:
    ```mlir
    // Apply Z to a single qubit
    %1 = quantum.Z %qubit : !quantum.qubit

    // Apply Z to each qubit in a 3-qubit register
    %2 = quantum.Z %register : !quantum.nqubit<3>
    ```
  }];

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_Qubit]>:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

//Custom R operation for x, y, z attributes
def Quantum_ROp : PrimitiveGate_Op<"R", [Pure]> {
  let summary = "Rotation gate operation around a principal axis (X, Y, or Z)";
  let description = [{
    The R gate applies a rotation around the X, Y, or Z axis by an angle θ.
    Example:
    ```mlir
    // Apply a rotation around the X-axis to a qubit with angle θ
    %1 = quantum.R(X, 2.4) %qubit : !quantum.qubit
    ```
  }];

  // Define the arguments: axis (as an enum), angle, and input qubit
  let arguments = (ins QuantumAxisAttr:$axis, F32:$theta, Quantum_Qubit:$input);

  // Define the result type
  let results = (outs Quantum_Qubit:$result);

  // Define the assembly format
  let assemblyFormat = [{
    `(`$input `,` $axis `,` $theta`)` attr-dict `:` type($input) `->` type($result)
  }];
}

// 1-qubit gates
def Quantum_SOp : PrimitiveGate_Op<"S", [Pure, SameOperandsAndResultType]> {
  let summary = "S gate operation (π/2 phase shift)";
  let description = [{
    The S gate applies a π/2 phase shift. It's equivalent to a Z rotation by π/2.
    Matrix representation: [[1, 0], [0, i]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($result)}];
}

def Quantum_TOp : PrimitiveGate_Op<"T", [Pure, SameOperandsAndResultType]> {
  let summary = "T gate operation (π/4 phase shift)";
  let description = [{
    The T gate applies a π/4 phase shift. It's equivalent to a Z rotation by π/4.
    Matrix representation: [[1, 0], [0, exp(iπ/4)]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($result) }];
}

def Quantum_SDaggerOp : PrimitiveGate_Op<"Sdagger", [Pure, SameOperandsAndResultType]> {
  let summary = "S dagger gate operation (-π/2 phase shift)";
  let description = [{
    The S dagger gate applies a -π/2 phase shift. It's the inverse of the S gate.
    Matrix representation: [[1, 0], [0, -i]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($result)}];
}

def Quantum_TDaggerOp : PrimitiveGate_Op<"Tdagger", [Pure, SameOperandsAndResultType]> {
  let summary = "T dagger gate operation (-π/4 phase shift)";
  let description = [{
    The T dagger gate applies a -π/4 phase shift. It's the inverse of the T gate.
    Matrix representation: [[1, 0], [0, exp(-iπ/4)]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict  `:` type($input) `->` type($result) }];
}

def Quantum_MeasureOp : Quantum_Op<"measure">{
      let summary = "Measure a single qubit.";
    let description = [{    }];

    let arguments = (ins Quantum_Qubit:$qinp);
    let results = (outs I1:$res, Quantum_Qubit:$qout);
    let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) `->` type($qout)}];
}

// 2-qubit gates
def Quantum_CYOp : PrimitiveGate_Op<"CY", [Pure]> {
  let summary = "Controlled-Y gate";
  let description = [{
    Applies a controlled-Y operation. The Y gate is applied to the target qubit
    if the control qubit is in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control `,` $target attr-dict `:`  type($control) `,` type($target) `->` type($control_out) `,` type($target_out) }];
}

def Quantum_CZOp : PrimitiveGate_Op<"CZ", [Pure]> {
  let summary = "Controlled-Z gate";
  let description = [{
    Applies a controlled-Z operation. The Z gate is applied to the target qubit
    if the control qubit is in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control `,` $target attr-dict `:`  type($control) `,` type($target) `->` type($control_out) `,` type($target_out) }];
}

def Quantum_SWAPOp : PrimitiveGate_Op<"SWAP", [Pure]> {
  let summary = "SWAP gate";
  let description = [{
    Swaps the states of two qubits.
  }];
  let arguments = (ins Quantum_Qubit:$qubit1, Quantum_Qubit:$qubit2);
  let results = (outs Quantum_Qubit:$qubit1_out, Quantum_Qubit:$qubit2_out);
  let assemblyFormat = [{ $qubit1 `,` $qubit2 attr-dict `:` type($qubit1) `,` type($qubit2) `->` type($qubit1_out) `,` type($qubit2_out)}];
}

def Quantum_CCXOp : PrimitiveGate_Op<"CCX", [Pure]> {
  let summary = "Controlled-Controlled-X (Toffoli) gate";
  let description = [{
    Applies a controlled-controlled-X (Toffoli) operation. The X gate is applied to the
    target qubit if both control qubits are in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control1, Quantum_Qubit:$control2, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control1_out, Quantum_Qubit:$control2_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control1 `,` $control2 `,` $target attr-dict `:`  type($control1) `,` type($control2) `,` type($target) `->` type($control1_out) `,` type($control2_out) `,` type($target_out)  }];
}

// n-qubit gates
def Quantum_CCZOp : PrimitiveGate_Op<"CCZ", [Pure]> {
  let summary = "Controlled-Controlled-Z gate";
  let description = [{
    Applies a controlled-controlled-Z operation. The Z gate is applied to the
    target qubit if both control qubits are in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control1, Quantum_Qubit:$control2, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control1_out, Quantum_Qubit:$control2_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control1 `,` $control2 `,` $target attr-dict `:`  type($control1) `,` type($control2) `,` type($target) `->` type($control1_out) `,` type($control2_out) `,` type($target_out) }];
}

def Quantum_CSWAPOp : PrimitiveGate_Op<"CSWAP", [Pure]> {
  let summary = "Controlled-SWAP (Fredkin) gate";
  let description = [{
    Applies a controlled-SWAP operation. The states of the two target qubits are
    swapped if the control qubit is in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control, Quantum_Qubit:$target1, Quantum_Qubit:$target2);
  let results = (outs Quantum_Qubit:$control_out, Quantum_Qubit:$target1_out, Quantum_Qubit:$target2_out);
  let assemblyFormat = [{ $control `,` $target1 `,` $target2 attr-dict `:`  type($control) `,` type($target1) `,` type($target2) `->` type($control_out) `,` type($target1_out) `,` type($target2_out) }];
}

// Universal gate
def Quantum_UOp : PrimitiveGate_Op<"U", [Pure]> {
  let summary = "Universal single-qubit rotation gate";
  let description = [{
    Applies a universal single-qubit rotation with three Euler angles.
    U(θ, φ, λ) = Rz(φ)Ry(θ)Rz(λ)
  }];
  let arguments = (ins F32:$theta, F32:$phi, F32:$lambda, Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input `(` $theta `,` $phi `,` $lambda `)` attr-dict `:`  type($input) `,` type($theta) `,` type($phi) `,` type($lambda) `->` type($result) }];
}

#endif
