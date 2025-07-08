//===- Ops.td - QPU dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the QPU dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef QPU_OPS
#define QPU_OPS

include "quantum-mlir/Dialect/QPU/IR/QPUTypes.td"
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
// QPU Op templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPU dialect traits.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// QPU operations.
//===----------------------------------------------------------------------===//

def QPU_QPUModuleOp : QPU_Op<"module", [
    IsolatedFromAbove,
    NoRegionArguments,
    SymbolTable,
    Symbol
  ]
  # GraphRegionNoTerminator.traits> {
  let summary = "Top level compilation unit containing code to be run on a QPU.";
  let description = [{
  
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<QPU_NonEmptyTargetArrayAttr>:$targets
  );
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name
    ($targets^)?
    attr-dict-with-keyword $bodyRegion
  }];
}

def QPU_CircuitOp : QPU_Op<"circuit", [
    IsolatedFromAbove,
    HasParent<"QPUModuleOp">,
    FunctionOpInterface
  ]> {
  let summary = "Quantum circuit operation with coupling graph";
  let description = [{ A quantum circuit operation that operates on a device with a specified coupling graph. }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region AnyRegion:$body);
  let builders = [OpBuilder<(ins
    "StringRef":$sym_name,
    "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    static CircuitOp create(Location location, StringRef name,
                         FunctionType type, ArrayRef<NamedAttribute> attrs = {});

    //===------------------------------------------------------------------===//
    // Methods to query the circuit function
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def QPU_ExecuteOp : QPU_Op<"execute", [
    AttrSizedOperandSegments
  ]> {
  let summary = "Executed a quantum circuit.";
  let description = [{
    Execute a quantum circuit. If the circuit has free parameters
    these must be provided to the execute operation.
  }];
  let arguments = (ins
    SymbolRefAttr:$circuit,
    Variadic<AnyType>:$circuitOperands,
    Variadic<AnyType>:$circuitResults
  );
  let assemblyFormat = [{
    $circuit
    custom<ExecuteOperands>($circuitOperands, type($circuitOperands))
    custom<ExecuteResults>($circuitResults, type($circuitResults))
    attr-dict
  }];
}

#endif // QPU_OPS
