//===- Attributes.td - Quantum dialect attributes --------*- tablegen -*-===//
//
// This is the definitions file for the Quantum dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_ATTRIBUTES
#define QUANTUM_ATTRIBUTES

include "cinm-mlir/Dialect/Quantum/IR/QuantumBase.td"

def Quantum_AxisX : I32EnumAttrCase<"X", 0, "x">;
def Quantum_AxisY : I32EnumAttrCase<"Y", 1, "y">;
def Quantum_AxisZ : I32EnumAttrCase<"Z", 2, "z">;

def QuantumAxis : I32EnumAttr<"QuantumAxis",
  "Axis of rotation for the quantum rotation operation", [
    Quantum_AxisX,
    Quantum_AxisY,
    Quantum_AxisZ
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::quantum";
}
def QuantumAxisAttr : EnumAttr<Quantum_Dialect, QuantumAxis, "quantum_axis">;

#endif
