// RUN: quantum-opt %s | FileCheck %s

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
  %const3 = arith.constant 0.23 : f64

  // QIR operations.
  "qir.U" (%q0, %const1, %const2, %const3) : (!qir.qubit, f64, f64, f64) -> ()
  "qir.H" (%q0) : (!qir.qubit) -> ()
  "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
  "qir.H" (%q1) : (!qir.qubit) -> ()
  "qir.H" (%q2) : (!qir.qubit) -> ()
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q1, %r1) : (!qir.qubit, !qir.result) -> ()
  "qir.measure" (%q2, %r2) : (!qir.qubit, !qir.result) -> ()

  return
}

// CHECK-DAG: module {
// CHECK-DAG:   func.func @main() {
// CHECK-DAG:     %0 = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG:     %1 = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG:     %2 = "qir.alloc"() : () -> !qir.qubit
// CHECK-DAG:     %3 = "qir.ralloc"() : () -> !qir.result
// CHECK-DAG:     %4 = "qir.ralloc"() : () -> !qir.result
// CHECK-DAG:     %5 = "qir.ralloc"() : () -> !qir.result
// CHECK-DAG:     %cst = arith.constant 3.400000e-01 : f64
// CHECK-DAG:     %cst_0 = arith.constant 7.350000e-01 : f64  
// CHECK-DAG:     %cst_1 = arith.constant 2.300000e-01 : f64
// CHECK-DAG:     "qir.U"(%0, %cst, %cst_0, %cst_1) : (!qir.qubit, f64, f64, f64) -> ()
// CHECK-DAG:     "qir.H"(%0) : (!qir.qubit) -> ()
// CHECK-DAG:     "qir.Rz"(%0, %cst) : (!qir.qubit, f64) -> ()
// CHECK-DAG:     "qir.H"(%1) : (!qir.qubit) -> ()
// CHECK-DAG:     "qir.H"(%2) : (!qir.qubit) -> ()
// CHECK-DAG:     "qir.measure"(%0, %3) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:     "qir.measure"(%1, %4) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:     "qir.measure"(%2, %5) : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG:     return
// CHECK-DAG:   }
// CHECK-DAG: }
