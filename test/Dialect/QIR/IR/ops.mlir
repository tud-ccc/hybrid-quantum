// RUN: quantum-opt %s --mlir-print-ir-after-all | FileCheck %s

func.func @main() {
  // Allocate qubits and results.
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  %q2 = "qir.alloc" () : () -> (!qir.qubit)

  %r0 = "qir.ralloc" () : () -> (!qir.result)
  %r1 = "qir.ralloc" () : () -> (!qir.result)
  %r2 = "qir.ralloc" () : () -> (!qir.result)

  // Create constants for rotation angles.
  %const1 = arith.constant 0.34 : f64
  %const2 = arith.constant 0.735 : f64

  // QIR operations.
  "qir.H" (%q0) : (!qir.qubit) -> ()
  "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
  "qir.H" (%q1) : (!qir.qubit) -> ()
  "qir.H" (%q2) : (!qir.qubit) -> ()
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q1, %r1) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q2, %r2) : (!qir.qubit, !qir.result) -> ()

  return
}

// CHECK: module {
// CHECK:   func.func @main() {
// CHECK:     %0 = "qir.alloc"() : () -> !qir.qubit
// CHECK:     %1 = "qir.alloc"() : () -> !qir.qubit
// CHECK:     %2 = "qir.alloc"() : () -> !qir.qubit
// CHECK:     %3 = "qir.ralloc"() : () -> !qir.result
// CHECK:     %4 = "qir.ralloc"() : () -> !qir.result
// CHECK:     %5 = "qir.ralloc"() : () -> !qir.result
// CHECK:     %cst = arith.constant 3.400000e-01 : f64
// CHECK:     %cst_0 = arith.constant 7.350000e-01 : f64
// CHECK:     "qir.H"(%0) : (!qir.qubit) -> ()
// CHECK:     "qir.Rz"(%0, %cst) : (!qir.qubit, f64) -> ()
// CHECK:     "qir.H"(%1) : (!qir.qubit) -> ()
// CHECK:     "qir.H"(%2) : (!qir.qubit) -> ()
// CHECK:     "qir.measure"(%0, %3) : (!qir.qubit, !qir.result) -> ()
// CHECK:     "qir.measure"(%1, %4) : (!qir.qubit, !qir.result) -> ()
// CHECK:     "qir.measure"(%2, %5) : (!qir.qubit, !qir.result) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
