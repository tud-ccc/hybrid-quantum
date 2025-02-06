//===- QuantumEnums.td --------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Quantum enumerations.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_ENUMS
#define QUANTUM_ENUMS

include "mlir/IR/EnumAttr.td"

def Quantum_NamedObservableAttr : I32EnumAttr<
    "NamedObservable",
    "Known named observables",
    [
        I32EnumAttrCase<"Identity", 0>,
        I32EnumAttrCase<"PauliX",   1>,
        I32EnumAttrCase<"PauliY",   2>,
        I32EnumAttrCase<"PauliZ",   3>,
        I32EnumAttrCase<"Hadamard", 4>]> {
    let cppNamespace = "::mlir::quantum";
}

def Quantum_NamedObservableDefaultAttr
        : DefaultValuedAttr<Quantum_NamedObservableAttr, "RoundingMode::Identity">;

def Quantum_QuantumAxis : I32EnumAttr<
  "QuantumAxis",
  "Axis of rotation for the quantum rotation operation",
  [
        I32EnumAttrCase<"X", 0, "x">,
        I32EnumAttrCase<"Y", 1, "y">,
        I32EnumAttrCase<"Z", 2, "z">]> {
  let cppNamespace = "::mlir::quantum";
}

//def QuantumAxisAttr : EnumAttr<Quantum_Dialect, Quantum_QuantumAxis, "quantum_axis">;

#endif // QUANTUM_ENUMS