// RUN: quantum-opt %s | FileCheck %s

func.func @main() {
  // Allocate qubits and results.
  %q0 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q0:.+]] = "qillr.alloc"()

  %q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q1:.+]] = "qillr.alloc"()

  %q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q2:.+]] = "qillr.alloc"()

  %r0 = "qillr.ralloc" () : () -> (!qillr.result)
// CHECK-DAG: %[[R0:.+]] = "qillr.ralloc"()

  %r1 = "qillr.ralloc" () : () -> (!qillr.result)
// CHECK-DAG: %[[R1:.+]] = "qillr.ralloc"()

  %r2 = "qillr.ralloc" () : () -> (!qillr.result)
// CHECK-DAG: %[[R2:.+]] = "qillr.ralloc"()

  // Create constants for rotation angles.
  %c1 = arith.constant 0.34  : f64
// CHECK-DAG: %[[C1:.+]] = arith.constant 3.400000e-01 : f64

  %c2 = arith.constant 0.735 : f64
// CHECK-DAG: %[[C2:.+]] = arith.constant 7.350000e-01 : f64

  %c3 = arith.constant 0.23  : f64
// CHECK-DAG: %[[C3:.+]] = arith.constant 2.300000e-01 : f64

  // Single-qubit Pauli & Hadamard gates
  "qillr.X"(%q0)   : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.X"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.Y"(%q0)   : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Y"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.Z"(%q0)   : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Z"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.H"(%q0)   : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.H"(%[[Q0]]) : (!qillr.qubit) -> ()

  // Rotation gates
  "qillr.Rx"(%q0, %c1) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.Rx"(%[[Q0]], %[[C1]]) : (!qillr.qubit, f64) -> ()

  "qillr.Ry"(%q0, %c2) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.Ry"(%[[Q0]], %[[C2]]) : (!qillr.qubit, f64) -> ()

  "qillr.Rz"(%q0, %c3) : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.Rz"(%[[Q0]], %[[C3]]) : (!qillr.qubit, f64) -> ()

  // Parameterized single-qubit gates
  "qillr.U1"(%q0, %c1)              : (!qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.U1"(%[[Q0]], %[[C1]]) : (!qillr.qubit, f64) -> ()

  "qillr.U2"(%q0, %c2, %c3)         : (!qillr.qubit, f64, f64) -> ()
// CHECK-DAG: "qillr.U2"(%[[Q0]], %[[C2]], %[[C3]]) : (!qillr.qubit, f64, f64) -> ()

  "qillr.U3" (%q0, %c1, %c2, %c3)    : (!qillr.qubit, f64, f64, f64) -> ()
// CHECK-DAG: "qillr.U3"(%[[Q0]], %[[C1]], %[[C2]], %[[C3]]) : (!qillr.qubit, f64, f64, f64) -> ()

  // Controlled and multi-qubit gates
  "qillr.CNOT"(%q0, %q1)            : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CNOT"(%[[Q0]], %[[Q1]]) : (!qillr.qubit, !qillr.qubit) -> ()

  "qillr.CRz"(%q0, %q1, %c1)        : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.CRz"(%[[Q0]], %[[Q1]], %[[C1]]) : (!qillr.qubit, !qillr.qubit, f64) -> ()

  "qillr.CRy"(%q0, %q1, %c2)        : (!qillr.qubit, !qillr.qubit, f64) -> ()
// CHECK-DAG: "qillr.CRy"(%[[Q0]], %[[Q1]], %[[C2]]) : (!qillr.qubit, !qillr.qubit, f64) -> ()

  "qillr.CCX"(%q0, %q1, %q2)        : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.CCX"(%[[Q0]], %[[Q1]], %[[Q2]]) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()

  // Phase gates
  "qillr.S"(%q0)                    : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.S"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.Sdg"(%q0)                  : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Sdg"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.T"(%q0)                    : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.T"(%[[Q0]]) : (!qillr.qubit) -> ()

  "qillr.Tdg"(%q0)                  : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.Tdg"(%[[Q0]]) : (!qillr.qubit) -> ()

  // Swap
  "qillr.swap"(%q0, %q1)            : (!qillr.qubit, !qillr.qubit) -> ()
// CHECK-DAG: "qillr.swap"(%[[Q0]], %[[Q1]]) : (!qillr.qubit, !qillr.qubit) -> ()

  // Measurements
  "qillr.measure"(%q0, %r0)         : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: "qillr.measure"(%[[Q0]], %[[R0]]) : (!qillr.qubit, !qillr.result) -> ()
  "qillr.measure"(%q1, %r1)         : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: "qillr.measure"(%[[Q1]], %[[R1]]) : (!qillr.qubit, !qillr.result) -> ()
  "qillr.measure"(%q2, %r2)         : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: "qillr.measure"(%[[Q2]], %[[R2]]) : (!qillr.qubit, !qillr.result) -> ()

  %m0 = "qillr.read_measurement"(%r0) : (!qillr.result) -> i1
// CHECK-DAG: %[[M0:.+]] = "qillr.read_measurement"(%[[R0]]) : (!qillr.result) -> i1

  // Reset
  "qillr.reset"(%q0)                : (!qillr.qubit) -> ()
// CHECK-DAG: "qillr.reset"(%[[Q0]]) : (!qillr.qubit) -> ()

  return
}
