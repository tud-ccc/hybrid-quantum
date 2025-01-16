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
def Quantum_AllocateOp : Memory_Op<"allocate", [NoMemoryEffect]> {
    let summary = "Allocate n qubits into a quantum register.";
    let description = [{
    }];

    let arguments = (ins
        Optional<I32>:$nqubits,
        OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$nqubits_attr
    );

    let results = (outs
        Quantum_QubitArray:$qreg
    );

    let assemblyFormat = [{
        `(` ($nqubits^):($nqubits_attr)? `)` attr-dict `:` type(results)
    }];
}


def Quantum_DeallocateOp : Memory_Op<"deallocate"> {
    let summary = "Deallocate a quantum register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_QubitArray:$qreg
    );

    let assemblyFormat = [{
        $qreg attr-dict `:` type(operands)
    }];

}

def Quantum_ExtractOp : Memory_Op<"extract", [NoMemoryEffect, Kernel]> {
    let summary = "Extract a qubit value from a register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_QubitArray:$qreg,
        Optional<I32>:$idx,
        OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$idx_attr
    );

    let results = (outs
        Quantum_Qubit:$qubit
    );

    let assemblyFormat = [{
        $qreg `[` ($idx^):($idx_attr)? `]` attr-dict `:` type($qreg) `->` type(results)
    }];

    let hasVerifier = 1;
}

def Quantum_InsertOp : Memory_Op<"insert", [NoMemoryEffect]> {
    let summary = "Update the qubit value of a register.";
    let description = [{
    }];

    let arguments = (ins
        Quantum_QubitArray:$in_qreg,
        Optional<I32>:$idx,
        OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$idx_attr,
        Quantum_Qubit:$qubit
    );

    let results = (outs
        Quantum_QubitArray:$out_qreg
    );

    let assemblyFormat = [{
        $in_qreg `[` ($idx^):($idx_attr)? `]` `,` $qubit attr-dict `:` type($in_qreg) `,` type($qubit)
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

  let arguments = (ins Quantum_QubitArray:$source,
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

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$result);

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

  let assemblyFormat = [{ $input `,` $target attr-dict }];
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

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$result);

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

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$result);

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

  let arguments = (ins AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$input);
  let results = (outs AnyTypeOf<[Quantum_Qubit, Quantum_QubitArray]>:$result);

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
    `(` $input attr-dict `,` $axis `,` $theta `)`
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
  let assemblyFormat = [{ $input attr-dict }];
}

def Quantum_TOp : PrimitiveGate_Op<"T", [Pure, SameOperandsAndResultType]> {
  let summary = "T gate operation (π/4 phase shift)";
  let description = [{
    The T gate applies a π/4 phase shift. It's equivalent to a Z rotation by π/4.
    Matrix representation: [[1, 0], [0, exp(iπ/4)]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict }];
}

def Quantum_SDaggerOp : PrimitiveGate_Op<"Sdagger", [Pure, SameOperandsAndResultType]> {
  let summary = "S dagger gate operation (-π/2 phase shift)";
  let description = [{
    The S dagger gate applies a -π/2 phase shift. It's the inverse of the S gate.
    Matrix representation: [[1, 0], [0, -i]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict }];
}

def Quantum_TDaggerOp : PrimitiveGate_Op<"Tdagger", [Pure, SameOperandsAndResultType]> {
  let summary = "T dagger gate operation (-π/4 phase shift)";
  let description = [{
    The T dagger gate applies a -π/4 phase shift. It's the inverse of the T gate.
    Matrix representation: [[1, 0], [0, exp(-iπ/4)]]
  }];
  let arguments = (ins Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input attr-dict }];
}

def Quantum_MeasureOp : PrimitiveGate_Op<"measure", [Pure]> {
    let summary = "Measure a single qubit.";
    let description = [{
        This operation takes a single qubit of type `qubit<1>`
        and returns the measurement outcome in the standard basis
        as a bit (`i1`)        ```
    }];
    let arguments = (ins Quantum_Qubit:$qinp);
    let results = (outs I1:$res, Quantum_Qubit:$qout);
    let assemblyFormat = [{ $qinp attr-dict }];
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
  let assemblyFormat = [{ $control `,` $target attr-dict }];
}

def Quantum_CZOp : PrimitiveGate_Op<"CZ", [Pure]> {
  let summary = "Controlled-Z gate";
  let description = [{
    Applies a controlled-Z operation. The Z gate is applied to the target qubit
    if the control qubit is in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control `,` $target attr-dict }];
}

def Quantum_SWAPOp : PrimitiveGate_Op<"SWAP", [Pure]> {
  let summary = "SWAP gate";
  let description = [{
    Swaps the states of two qubits.
  }];
  let arguments = (ins Quantum_Qubit:$qubit1, Quantum_Qubit:$qubit2);
  let results = (outs Quantum_Qubit:$qubit1_out, Quantum_Qubit:$qubit2_out);
  let assemblyFormat = [{ $qubit1 `,` $qubit2 attr-dict }];
}

def Quantum_CCXOp : PrimitiveGate_Op<"CCX", [Pure]> {
  let summary = "Controlled-Controlled-X (Toffoli) gate";
  let description = [{
    Applies a controlled-controlled-X (Toffoli) operation. The X gate is applied to the
    target qubit if both control qubits are in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control1, Quantum_Qubit:$control2, Quantum_Qubit:$target);
  let results = (outs Quantum_Qubit:$control1_out, Quantum_Qubit:$control2_out, Quantum_Qubit:$target_out);
  let assemblyFormat = [{ $control1 `,` $control2 `,` $target attr-dict }];
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
  let assemblyFormat = [{ $control1 `,` $control2 `,` $target attr-dict }];
}

def Quantum_CSWAPOp : PrimitiveGate_Op<"CSWAP", [Pure]> {
  let summary = "Controlled-SWAP (Fredkin) gate";
  let description = [{
    Applies a controlled-SWAP operation. The states of the two target qubits are
    swapped if the control qubit is in the |1⟩ state.
  }];
  let arguments = (ins Quantum_Qubit:$control, Quantum_Qubit:$target1, Quantum_Qubit:$target2);
  let results = (outs Quantum_Qubit:$control_out, Quantum_Qubit:$target1_out, Quantum_Qubit:$target2_out);
  let assemblyFormat = [{ $control `,` $target1 `,` $target2 attr-dict }];
}

// Universal gate
def Quantum_UOp : PrimitiveGate_Op<"U", [Pure, SameOperandsAndResultType]> {
  let summary = "Universal single-qubit rotation gate";
  let description = [{
    Applies a universal single-qubit rotation with three Euler angles.
    U(θ, φ, λ) = Rz(φ)Ry(θ)Rz(λ)
  }];
  let arguments = (ins F32Attr:$theta, F32Attr:$phi, F32Attr:$lambda, Quantum_Qubit:$input);
  let results = (outs Quantum_Qubit:$result);
  let assemblyFormat = [{ $input `(` $theta `,` $phi `,` $lambda `)` attr-dict }];
}

//observation ops
def Quantum_ExpectationOp : Measurement_Op<"expectation"> {
    let summary = "Compute the expectation value of the given observable for the current state";
    let description = [{
        Example:

        ```mlir
        func.func @foo(%q: !quantum.bit)
        {
            %obs = quantum.namedobs %q[4] : !quantum.obs
            %expval = quantum.expectation %obs : f64

            func.return
        }
        ```
    }];

    let arguments = (ins
        Quantum_ObservableType:$input
    );

    let results = (outs
        F32:$expval
    );

    let assemblyFormat = [{
        $input attr-dict `:` type(results)
    }];
}

def Quantum_HamiltonianOp : Observable_Op<"hamiltonian"> {
    let summary = "Define a Hamiltonian observable for use in measurements";
    let description = [{
        ```mlir
        func.func @foo(%q0: !quantum.bit, %q1: !quantum.bit, %c: tensor<2xf64>) {
            %obs1 = quantum.namedobs %q0[4] : !quantum.obs
            %obs2 = quantum.namedobs %q1[2] : !quantum.obs
            %res = quantum.hamiltonian(%c: tensor<2xf64>) %obs1, %obs2 : !quantum.obs
            func.return
        }
        ```
    }];

    let arguments = (ins
        AnyTypeOf<[
            1DTensorOf<[F32]>, MemRefRankOf<[F32], [1]>
        ]>:$coeffs,
        Quantum_Qubit:$input,
        I32:$ntype
    );

    let results = (outs
        Quantum_ObservableType:$out
    );

    let assemblyFormat = [{
         $input `,` $coeffs `,` $ntype attr-dict `:` type($input)  `,` type($coeffs) `,`  type($ntype)  `->` type(results)
    }];
}


def Quantum_CircuitOp : Quantum_Op<"circuit", [
  AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, Symbol,
  RecursiveMemoryEffects
]> {
  let summary = "An operation with a name containing a single `SSACFG` region corresponding to a quantum circuit(block) execution";
  let description = [{

    Defines a circuit(function) operation representing a
    proecedural (deterministically timed) circuit execution
    on the QPU *without* any classical compute or control-flow.
    Effectively corresponding to a callable "circuit block".

    The CircuitOp is modelled after the FuncOp of the MLIR func
    dialect.

    A circuit may only contain operations declaring qubits
    TODO: move qubit declaration/allocation outside of the circuit),
    unitary gates operations, and projective measurements. It may only
    be launched from the QPU's control-system directly.

    All non-constant classical values required to define the circuit must
    be provided as circuit arguments to allow a clean separation of
    classical and quantum processes within the MLIR.

    Quantum programs should be canonicalized such that all quantum
    operations occurr within a region contained by a `quir.circuit` operation
    and are reached through a corresponding `quir.call_circuit`.

    TODO: Clarify the difference between circuit and gate definitions.

    Operations within the circuit cannot implicitly capture values defined
    outside of the circuit, i.e. they are `IsolatedFromAbove`. All
    external references must use circuit arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external circuit declaration (used when
    referring to a circuit declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for circuit arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Syntax:

    ```
    op ::= `quir.circuit` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)? function-attributes? region
    ```

    Example:

    ```mlir
    // External circuit function definitions.
    quir.circuit @x()

    // A function that returns its argument twice:
    quir.circuit @vqe(%angle: quir.angle<32>) -> i1
      {quir.physicalId = 0 : i32} {
      %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      quir.gate_func.call @rz(%0, %angle) : (!quir.qubit<1>, !quir.angle<32>) -> ()
      %result = "quir.measure"(%0) : (!quir.qubit<1>) -> i1
      quir.return %result: i1
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];
  let extraClassDeclaration = [{
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this circuit and all of its blocks, remapping any
    /// operands that use values outside of the circuit using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for circuit arguments, these arguments are not
    /// included in the new circuit. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    CircuitOp clone(IRMapping &mapper);
    CircuitOp clone();

    /// Clone the internal blocks and attributes from this circuit into dest.
    /// Any cloned blocks are appended to the back of dest. This circuit
    /// asserts that the attributes of the current circuit and dest are
    /// compatible.
    void cloneInto(CircuitOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Quantum_ReturnOp : Quantum_Op<"return", [HasParent<"CircuitOp">, Pure, Terminator, ReturnLike]>,
    Arguments<(ins Variadic<AnyType>:$operands)>, Results<(outs)> {
        let summary = "Terminator for quir.circuit.";
        let description = [{
                A terminator for regions that appear in the body of the `quir.circuit`
                circuit operation. The operands to the `quir.return` are the result values
                that are returned to the `quir.call_circuit` invocation of the target circuit.

                Example:

                ```mlir
                quir.circuit @foo(%angle: quir.angle<32>) -> (i1, i1)
                    ...
                    quir.return %0, %1: i1, i1
                }
            }];
        let builders = [OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>];
        let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
        let hasVerifier = 1;
}

def Quantum_CallCircuitOp : Quantum_Op<"call_circuit", [CallOpInterface, MemRefsNormalizable, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Call a circuit operation";
    let description = [{
        The `quir.call_circuit` operation represents calls to launch a quantum circuit in the target system.
        The callee function is attached to a symbol reference via an attribute. The arguments list must match
        the argument types provided by the callee. The calling of a circuit represents the classical<->quantum
        interaction within the specified program. All classical input values to the circuit should be transferred
        before the invocation of the circuit routine to enable deterministic execution of the quantum circuit.

        TODO: Currently all qubits must be declared within the circuit for lowering to hardware due to how
        qubit allocations are currently tracked with `declare_qubit`. In the future we should consider
        supporting quantum arguments (qubits) to make truly reuseable circuit routines that align
        with the Qiskit and OpenQASM3 circuit definitions.

        Example:
        ```mlir
        %classical_result = quir.call_circuit @vqe(%theta) : (quir.angle<32>) -> (i1, i1)
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "CircuitOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
    }];
}



#endif
