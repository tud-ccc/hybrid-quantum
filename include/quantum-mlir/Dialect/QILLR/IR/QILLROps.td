//===- Ops.td - QILLR dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the QILLR dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QILLR_OPS
#define QILLR_OPS

include "mlir/IR/EnumAttr.td"
//include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "quantum-mlir/Dialect/QILLR/IR/QILLRBase.td"
include "quantum-mlir/Dialect/QILLR/IR/QILLRTypes.td"
include "mlir/IR/BuiltinTypes.td"



//===----------------------------------------------------------------------===//
// QILLR device operations.
//===----------------------------------------------------------------------===//
def QILLR_InitOp : QILLR_Op<"init">{
  let summary = "Allocates a qubit.";
  let description = [{  }];
}

def QILLR_SeedOp : QILLR_Op<"seed">{
  let summary = "Set a seed for deterministic measurements.";
  let description = [{  }];
 let arguments = (ins I64:$seed);
}

//===----------------------------------------------------------------------===//
// QILLR memory operations.
//===----------------------------------------------------------------------===//
def QILLR_AllocOp : Memory_Op<"alloc", [MemoryEffects<[MemRead]>]> {
  let summary = "Allocates a qubit.";
  let description = [{  }];

  let results = (outs QILLR_QubitType:$result);
}

def QILLR_AllocResultOp : Memory_Op<"ralloc", [MemoryEffects<[MemRead]>]> {
  let summary = "Allocate n results into a result register.";
  let description = [{ Allocates a fixed number of results into a result register. }];

  let results = (outs QILLR_ResultType:$result);
}

def QILLR_ShowStateOp : Memory_Op<"show_state", [MemoryEffects<[MemRead]>]> {
  let summary = "Show the internal state of the simulator.";
  let description = [{  }];
}

//===----------------------------------------------------------------------===//
// Single qubit gate operations.
//===----------------------------------------------------------------------===//
class PrimitiveGate_Op<string mnemonic, list<Trait> traits = []>:Gate_Op<mnemonic, traits>{
  // Declare a mnemonic field so that derived definitions can override it.
  let arguments = (ins QILLR_QubitType:$input);
}
        
def QILLR_HOp : PrimitiveGate_Op<"H", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Hadamard gate operation";
  let description = [{
    The Hadamard gate creates an equal superposition of states.
  }];
}


def QILLR_XOp : PrimitiveGate_Op<"X"> {
  let summary = "X gate operation";
  let description = [{
    The X gate negates a qubit value.
  }];
}

def QILLR_ZOp : PrimitiveGate_Op<"Z"> {
  let summary = "Z gate operation";
  let description = [{
    The Z gate negates a qubit value.
  }];
}

def QILLR_YOp : PrimitiveGate_Op<"Y"> {
  let summary = "Y gate operation";
  let description = [{
    The Y gate negates a qubit value.
  }];
}


def QILLR_SOp : PrimitiveGate_Op<"S"> {
  let summary = "S gate operation";
  let description = [{ Phase gate (√Z). }];
}

def QILLR_SXOp : PrimitiveGate_Op<"SX"> {
  let summary = "SX gate operation";
  let description = [{ SX. }];
} 

def QILLR_SdgOp : PrimitiveGate_Op<"Sdg"> {
  let summary = "S-dagger gate operation";
  let description = [{ Adjoint of the S gate. }];
}

def QILLR_TOp : PrimitiveGate_Op<"T"> {
  let summary = "T gate operation";
  let description = [{ T gate (π/8 gate). }];
}

def QILLR_TdgOp : PrimitiveGate_Op<"Tdg"> {
  let summary = "T-dagger gate operation";
  let description = [{ Adjoint of the T gate. }];
}

def QILLR_IdOp : PrimitiveGate_Op<"Id"> {
  let summary = "Identity gate operation";
  let description = [{ Identitiy. }];
}

//===----------------------------------------------------------------------===//
// Two qubit gate operations.
//===----------------------------------------------------------------------===//
def QILLR_CNOTOp : Gate_Op<"CNOT"> {
  let summary = "CNOT gate operation";
  let description = [{
    The controlled-NOT (CNOT) gate flips the target qubit if the control qubit is in the |1> state.
  }];
  let arguments = (ins QILLR_QubitType:$control, QILLR_QubitType:$target);
}

def QILLR_CZOp : Gate_Op<"Cz"> {
  let summary = "CZ gate operation";
  let description = [{
    The controlled-Z (CZ) gate
  }];
  let arguments = (ins QILLR_QubitType:$control, QILLR_QubitType:$target);
}

def QILLR_SwapOp : Gate_Op<"swap"> {
  let summary = "Swap gate operation";
  let description = [{
    The swap gate exchanges the quantum states of two qubits.
  }];
  let arguments = (ins
    QILLR_QubitType:$lhs,
    QILLR_QubitType:$rhs
  );
}

//===----------------------------------------------------------------------===//
// Parametric single qubit gate operations.
//===----------------------------------------------------------------------===//
def QILLR_RxOp : Gate_Op<"Rx"> {
  let summary = "Rx gate operation";
  let description = [{
    The X gate negates a qubit value.
  }];

  let arguments = (ins QILLR_QubitType:$input, F64:$angle);
}

// Rz: a single-qubit rotation with an angle parameter.
def QILLR_RzOp : Gate_Op<"Rz"> {
  let summary = "Rz gate operation";
  let description = [{
    The Rz gate rotates a qubit around the Z-axis by a specified angle.
  }];
  let arguments = (ins QILLR_QubitType:$input, F64:$angle);
}

// Ry: a single-qubit rotation with an angle parameter.
def QILLR_RyOp : Gate_Op<"Ry"> {
  let summary = "Ry gate operation";
  let description = [{
    The Ry gate rotates a qubit around the Y-axis by a specified angle.
  }];
  let arguments = (ins QILLR_QubitType:$input, F64:$angle);
}

def QILLR_PhaseOp : Gate_Op<"P"> {
  let summary = "Phase gate operation";
  let description = [{ Phase gate with rotation around Z axis. }];
  let arguments = (ins QILLR_QubitType:$input, F64:$angle);
} 

// U: a universal single-qubit gate with three angles.
def QILLR_U3Op : Gate_Op<"U3"> {
  let summary = "U gate operation";
  let description = [{
    The U gate is a universal single-qubit gate that can represent any single-qubit operation.
  }];
  let arguments = (ins QILLR_QubitType:$input, F64:$theta, F64:$phi, F64:$lambda);
}

def QILLR_U2Op : Gate_Op<"U2"> {
  let summary = "U2 gate operation";
  let description = [{ A single-qubit gate with two parameters φ and λ. Equivalent to U3(π/2, φ, λ). }];
  let arguments = (ins QILLR_QubitType:$input, F64:$phi, F64:$lambda);
}

def QILLR_U1Op : Gate_Op<"U1"> {
  let summary = "U1 gate operation";
  let description = [{ A single-parameter unitary gate for phase rotation. Equivalent to Rz(λ). }];
  let arguments = (ins QILLR_QubitType:$input, F64:$lambda);
}


//===----------------------------------------------------------------------===//
// Controlled Parametric two qubit gate operations.
//===----------------------------------------------------------------------===//
def QILLR_CRzOp : Gate_Op<"CRz"> {
  let summary = "Controlled-Rz gate";
  let description = [{ Applies a Z rotation conditioned on the control qubit. }];
  let arguments = (ins QILLR_QubitType:$control, QILLR_QubitType:$target, F64:$angle);
}

def QILLR_CRyOp : Gate_Op<"CRy"> {
  let summary = "Controlled-Ry gate";
  let description = [{ Applies a Y rotation conditioned on the control qubit. }];
  let arguments = (ins QILLR_QubitType:$control, QILLR_QubitType:$target, F64:$angle);
}

def QILLR_CU1Op : Gate_Op<"CU1"> {
  let summary = "Controlled-U1 gate";
  let description = [{
    The controlled-U1 (CU1) gate
  }];
  let arguments = (ins QILLR_QubitType:$control, QILLR_QubitType:$target, F64:$angle);
}


//===----------------------------------------------------------------------===//
// Multiqubit/Universal/Custom gate operations.
//===----------------------------------------------------------------------===//
def QILLR_CCXOp : Gate_Op<"CCX"> {
  let summary = "Toffoli (CCX) gate";
  let description = [{ Controlled-controlled-X gate. Flips target if both controls are |1>. }];
  let arguments = (ins QILLR_QubitType:$control1, QILLR_QubitType:$control2, QILLR_QubitType:$target);
}

def QILLR_CSwapOp : Gate_Op<"cswap"> {
  let summary = "Controlled-swap gate operation";
  let description = [{
    The controlled-swap gate exchanges the quantum states of two qubits.
  }];
  let arguments = (ins
    QILLR_QubitType:$control,
    QILLR_QubitType:$lhs,
    QILLR_QubitType:$rhs
  );
}

def QILLR_BarrierOp : Gate_Op<"barrier"> {
  let summary = "Barrier operation";
  let description = [{ A barrier operation that prevents optimization across it. }];
  let arguments = (ins Variadic<QILLR_QubitType>:$input);
}

def QILLR_GateCallOp : Gate_Op<"call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Gate call operation";
  let description = [{ Gate call operation. }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );

  let builders = [
    OpBuilder<(ins "GateOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>
  ];

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
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];
}

def QILLR_GateOp : Gate_Op<"gate", [
  IsolatedFromAbove,
  FunctionOpInterface
]> {
  let summary = "Custom gate operation";
  let description = [{ A custom gate operation. }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    TypeAttrOf<FunctionType>:$function_type
  );

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    static GateOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    
    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return false; }
  }];
}

def QILLR_ReturnOp : QILLR_Op<"return", [
  Pure,
  HasParent<"GateOp">,
  ReturnLike,
  Terminator]> {
  let summary = "QILLR gate return operation";
  let description = [{
    The `QILLR.return` operation represents a return operation within a gate body.
    The operation takes no operands and produces no results.
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Measurement related gate operations.
//===----------------------------------------------------------------------===//
def QILLR_MeasureOp : QILLR_Op<"measure", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "Measure a single qubit.";
    let description = [{ }];

    let arguments = (ins
      QILLR_QubitType:$input,
      QILLR_ResultType:$result
    );
}

def QILLR_ReadMeasurementOp : QILLR_Op<"read_measurement", [MemoryEffects<[MemRead]>]> {
  let summary = "Read the measurement value from result memory";
  let description = [{ }];
  
  let arguments = (ins
    QILLR_ResultType:$input
  );
  let results = (outs
    I1:$measurement
  );
}

def QILLR_ResetOp : QILLR_Op<"reset", [MemoryEffects<[MemRead]>]> {
  let summary = "Resets the qubit in the computational basis";
  let description = [{ }];
  
  let arguments = (ins QILLR_QubitType:$input);
}


#endif
