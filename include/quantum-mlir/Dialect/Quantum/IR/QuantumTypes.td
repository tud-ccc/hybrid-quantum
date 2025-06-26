//===- Types.td - Quantum dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "quantum-mlir/Dialect/Quantum/IR/QuantumAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Quantum_QubitType : Quantum_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "Multi-qubit register type";
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
  let extraClassDeclaration = [{
    bool isSingleQubit() const;
  }];
  let genVerifyDecl = 1;
}

class QubitSizePred<int N> : CPred<"$_self.cast<QubitType>().getSize() == " # N>;

class QubitArrayOfSize<int N>
  : Type<And<[Quantum_QubitType.predicate, QubitSizePred<N>]>,
         "Quantum " # N # "-Qubit Type">,
    BuildableType<"QubitType::get($_builder.getContext(), " # N # ")">;

def Quantum_SingleQubitType : QubitArrayOfSize<1>;

def Quantum_MeasurementType : 1DTensorOf<[I1]>;

def Quantum_DeviceType : Quantum_Type<"Device"> {
  let mnemonic = "device";
  let description = "Quantum device type that holds information about the device size and coupling graph.";
  let parameters = (ins
    "int64_t":$qubits,
    "ArrayAttr":$edges
  );
  //let assemblyFormat = "`<` $qubits `,` $edges `>`";
  let extraClassDeclaration = [{
    bool hasStaticSize() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def Quantum_CircuitType : Quantum_Type<"Circuit"> {
  let mnemonic = "circuit";
  let description = "Quantum circuit type refers to a circuit that contains quantum operations.";
}

#endif
