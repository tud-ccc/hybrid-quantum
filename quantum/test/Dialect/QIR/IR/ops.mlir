// RUN: quantum-opt %s --convert-qir-to-llvm | FileCheck %s

func.func @main() {
  // Allocate qubits and results.
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  %q2 = "qir.alloc" () : () -> (!qir.qubit)

  %r0 = "qir.ralloc" () : () -> (!qir.result)
  %r1 = "qir.ralloc" () : () -> (!qir.result)
  %r2 = "qir.ralloc" () : () -> (!qir.result)

  // Create constants for rotation angles.
  %const1 = arith.constant 0.34 : f32
  %const2 = arith.constant 0.735 : f32

  // QIR operations.
  "qir.H" (%q0) : (!qir.qubit) -> ()
  "qir.Rz" (%q0, %const1) : (!qir.qubit, f32) -> ()
  "qir.H" (%q1) : (!qir.qubit) -> ()
  "qir.H" (%q2) : (!qir.qubit) -> ()
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q1, %r1) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q2, %r2) : (!qir.qubit, !qir.result) -> ()

  return
}

// CHECK-LABEL: func.func @main()
// CHECK: %q0 = "qir.alloc" () : () -> (!qir.qubit)
// CHECK: %const1 = arith.constant 0.34 : f32
// CHECK: call void @__quantum__qis__h__body(%q0)
// CHECK: call void @__quantum__qis__rz__body(%const1, %q0)
// CHECK: call void @__quantum__qis__h__body(%q1)
// CHECK: call void @__quantum__qis__h__body(%q2)
// CHECK: call void @__quantum__qis__measure__body(%q0, %r0)
// CHECK: call void @__quantum__qis__measure__body(%q1, %r1)
// CHECK: call void @__quantum__qis__measure__body(%q2, %r2)
