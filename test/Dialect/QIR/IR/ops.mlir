// RUN: quantum-opt %s | FileCheck %s

func.func @main() {
  // Allocate qubits and results.
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: %[[Q0:.+]] = "qir.alloc"()

  %q1 = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: %[[Q1:.+]] = "qir.alloc"()

  %q2 = "qir.alloc" () : () -> (!qir.qubit)
// CHECK-DAG: %[[Q2:.+]] = "qir.alloc"()

  %r0 = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG: %[[R0:.+]] = "qir.ralloc"()

  %r1 = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG: %[[R1:.+]] = "qir.ralloc"()

  %r2 = "qir.ralloc" () : () -> (!qir.result)
// CHECK-DAG: %[[R2:.+]] = "qir.ralloc"()

  // Create constants for rotation angles.
  %c1 = arith.constant 0.34  : f64
// CHECK-DAG: %[[C1:.+]] = arith.constant 3.400000e-01 : f64

  %c2 = arith.constant 0.735 : f64
// CHECK-DAG: %[[C2:.+]] = arith.constant 7.350000e-01 : f64

  %c3 = arith.constant 0.23  : f64
// CHECK-DAG: %[[C3:.+]] = arith.constant 2.300000e-01 : f64

  // Single-qubit Pauli & Hadamard gates
  "qir.X"(%q0)   : (!qir.qubit) -> ()
// CHECK-DAG: "qir.X"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.Y"(%q0)   : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Y"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.Z"(%q0)   : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Z"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.H"(%q0)   : (!qir.qubit) -> ()
// CHECK-DAG: "qir.H"(%[[Q0]]) : (!qir.qubit) -> ()

  // Rotation gates
  "qir.Rx"(%q0, %c1) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.Rx"(%[[Q0]], %[[C1]]) : (!qir.qubit, f64) -> ()

  "qir.Ry"(%q0, %c2) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.Ry"(%[[Q0]], %[[C2]]) : (!qir.qubit, f64) -> ()

  "qir.Rz"(%q0, %c3) : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.Rz"(%[[Q0]], %[[C3]]) : (!qir.qubit, f64) -> ()

  // Parameterized single-qubit gates
  "qir.U1"(%q0, %c1)              : (!qir.qubit, f64) -> ()
// CHECK-DAG: "qir.U1"(%[[Q0]], %[[C1]]) : (!qir.qubit, f64) -> ()

  "qir.U2"(%q0, %c2, %c3)         : (!qir.qubit, f64, f64) -> ()
// CHECK-DAG: "qir.U2"(%[[Q0]], %[[C2]], %[[C3]]) : (!qir.qubit, f64, f64) -> ()

  "qir.U3" (%q0, %c1, %c2, %c3)    : (!qir.qubit, f64, f64, f64) -> ()
// CHECK-DAG: "qir.U3"(%[[Q0]], %[[C1]], %[[C2]], %[[C3]]) : (!qir.qubit, f64, f64, f64) -> ()

  // Controlled and multi-qubit gates
  "qir.CNOT"(%q0, %q1)            : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CNOT"(%[[Q0]], %[[Q1]]) : (!qir.qubit, !qir.qubit) -> ()

  "qir.CRz"(%q0, %q1, %c1)        : (!qir.qubit, !qir.qubit, f64) -> ()
// CHECK-DAG: "qir.CRz"(%[[Q0]], %[[Q1]], %[[C1]]) : (!qir.qubit, !qir.qubit, f64) -> ()

  "qir.CRy"(%q0, %q1, %c2)        : (!qir.qubit, !qir.qubit, f64) -> ()
// CHECK-DAG: "qir.CRy"(%[[Q0]], %[[Q1]], %[[C2]]) : (!qir.qubit, !qir.qubit, f64) -> ()

  "qir.CCX"(%q0, %q1, %q2)        : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.CCX"(%[[Q0]], %[[Q1]], %[[Q2]]) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()

  // Phase gates
  "qir.S"(%q0)                    : (!qir.qubit) -> ()
// CHECK-DAG: "qir.S"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.Sdg"(%q0)                  : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Sdg"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.T"(%q0)                    : (!qir.qubit) -> ()
// CHECK-DAG: "qir.T"(%[[Q0]]) : (!qir.qubit) -> ()

  "qir.Tdg"(%q0)                  : (!qir.qubit) -> ()
// CHECK-DAG: "qir.Tdg"(%[[Q0]]) : (!qir.qubit) -> ()

  // Swap
  "qir.swap"(%q0, %q1)            : (!qir.qubit, !qir.qubit) -> ()
// CHECK-DAG: "qir.swap"(%[[Q0]], %[[Q1]]) : (!qir.qubit, !qir.qubit) -> ()

  // Measurements
  "qir.measure"(%q0, %r0)         : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG: "qir.measure"(%[[Q0]], %[[R0]]) : (!qir.qubit, !qir.result) -> ()
  "qir.measure"(%q1, %r1)         : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG: "qir.measure"(%[[Q1]], %[[R1]]) : (!qir.qubit, !qir.result) -> ()
  "qir.measure"(%q2, %r2)         : (!qir.qubit, !qir.result) -> ()
// CHECK-DAG: "qir.measure"(%[[Q2]], %[[R2]]) : (!qir.qubit, !qir.result) -> ()

  %m0 = "qir.read_measurement"(%r0) : (!qir.result) -> (tensor<1xi1>)
// CHECK-DAG: %[[M0:.+]] = "qir.read_measurement"(%[[R0]]) : (!qir.result) -> tensor<1xi1>

  // Reset
  "qir.reset"(%q0)                : (!qir.qubit) -> ()
// CHECK-DAG: "qir.reset"(%[[Q0]]) : (!qir.qubit) -> ()

  return
}
