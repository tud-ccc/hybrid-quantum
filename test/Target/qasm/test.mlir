// RUN: quantum-translate --mlir-to-openqasm %s | FileCheck %s

module {
  // Initialization and diagnostics
  // CHECK-DAG: OPENQASM 2.0;
  // CHECK-DAG: include "qelib1.inc";

  // CHECK-DAG: qreg q0[1];
  // CHECK-DAG: qreg q1[1];
  // CHECK-DAG: qreg q2[1];
  %q0 = "qir.alloc"() : () -> (!qir.qubit)
  %q1 = "qir.alloc"() : () -> (!qir.qubit)
  %q2 = "qir.alloc"() : () -> (!qir.qubit)

  // Some constant parameters for rotations and U gates
  %c1 = arith.constant 0.1 : f64
  %c2 = arith.constant 0.2 : f64
  %c3 = arith.constant 0.3 : f64

  // Allocate two result registers for measurements
  // CHECK-DAG: creg c0[1];
  // CHECK-DAG: creg c1[1];
  %r0 = "qir.ralloc"() : () -> (!qir.result)
  %r1 = "qir.ralloc"() : () -> (!qir.result)

  // Basic gates and operations
  // CHECK-DAG: h q0;
  "qir.H"(%q0)      : (!qir.qubit) -> ()
  // CHECK-DAG: x q1;
  "qir.X"(%q1)      : (!qir.qubit) -> ()
  // CHECK-DAG: u3({{.*}}) q2;
  "qir.U3"(%q2, %c1, %c2, %c3) : (!qir.qubit, f64, f64, f64) -> ()
  // CHECK-DAG: cx q0, q1;
  "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()

  // Extended single-qubit gates
  // CHECK-DAG: z q0;
  "qir.Z"(%q0)      : (!qir.qubit) -> ()
  // CHECK-DAG: y q1;
  "qir.Y"(%q1)      : (!qir.qubit) -> ()
  // CHECK-DAG: rx({{.*}}) q2;
  "qir.Rx"(%q2, %c1) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: ry({{.*}}) q0;
  "qir.Ry"(%q0, %c2) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: rz({{.*}}) q1;
  "qir.Rz"(%q1, %c3) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: u2({{.*}}) q2;
  "qir.U2"(%q2, %c1, %c2) : (!qir.qubit, f64, f64) -> ()
  // CHECK-DAG: u1({{.*}}) q0;
  "qir.U1"(%q0, %c3) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: s q1;
  "qir.S"(%q1)      : (!qir.qubit) -> ()
  // CHECK-DAG: sdg q2;
  "qir.Sdg"(%q2)    : (!qir.qubit) -> ()
  // CHECK-DAG: t q0;
  "qir.T"(%q0)      : (!qir.qubit) -> ()
  // CHECK-DAG: tdg q1;
  "qir.Tdg"(%q1)    : (!qir.qubit) -> ()

  // Extended multi-qubit gates
  // CHECK-DAG: cz q0, q2;
  "qir.Cz"(%q0, %q2)   : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: crz({{.*}}) q1, q0;
  "qir.CRz"(%q1, %q0, %c3) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-DAG: cry({{.*}}) q2, q1;
  "qir.CRy"(%q2, %q1, %c2) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-DAG: ccx q0, q1, q2;
  "qir.CCX"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: barrier q0, q1, q2;
  "qir.barrier"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: swap q1, q2;
  "qir.swap"(%q1, %q2) : (!qir.qubit, !qir.qubit) -> ()

  // Measurement utilities
  // CHECK-DAG: measure q2 -> c1[0];
  "qir.measure"(%q2, %r1) : (!qir.qubit, !qir.result) -> ()
  // CHECK-DAG: // read_measurement into c1
  %mread = "qir.read_measurement"(%r1) : (!qir.result) -> (tensor<1xi1>)
  // CHECK-DAG: reset q2;
  "qir.reset"(%q2) : (!qir.qubit) -> ()
}
