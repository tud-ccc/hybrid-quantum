// RUN: quantum-translate --mlir-to-openqasm %s | FileCheck %s

module {
  // CHECK: OPENQASM 2.0;
  // CHECK-NEXT: include "qelib1.inc";
  // CHECK-DAG: qreg [[q0:reg.+]][1]; 
  %q0 = "qir.alloc"() : () -> (!qir.qubit)
  // CHECK-NEXT: qreg [[q1:reg.+]][1];
  %q1 = "qir.alloc"() : () -> (!qir.qubit)
  // CHECK-NEXT: qreg [[q2:reg.+]][1];
  %q2 = "qir.alloc"() : () -> (!qir.qubit)
  // CHECK-NEXT: creg [[c0:reg.+]][1];
  %r0 = "qir.ralloc"() : () -> (!qir.result)
  // CHECK-NEXT: creg [[c1:reg.+]][1];
  %r1 = "qir.ralloc"() : () -> (!qir.result)
  // Basic gates and operations
  // CHECK-NEXT: h [[q0]];
  "qir.H"(%q0): (!qir.qubit) -> ()
  // CHECK-NEXT: x [[q1]];
  "qir.X"(%q1): (!qir.qubit) -> ()
  // CHECK-NOT: arith.constant
  %c1 = arith.constant 0.1 : f64
  %c2 = arith.constant 0.2 : f64
  %c3 = arith.constant 0.3 : f64
  // CHECK-NEXT: u3({{.+}},{{.+}},{{.+}}) [[q2]];
  "qir.U3"(%q2, %c1, %c2, %c3) : (!qir.qubit, f64, f64, f64) -> ()
  // CHECK-NEXT: cx [[q0]], [[q1]];
  "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-NEXT: z [[q0]];
  "qir.Z"(%q0)      : (!qir.qubit) -> ()
  // CHECK-NEXT: y [[q1]];
  "qir.Y"(%q1)      : (!qir.qubit) -> ()
  // CHECK-NEXT: rx({{.*}}) [[q2]];
  "qir.Rx"(%q2, %c1) : (!qir.qubit, f64) -> ()
  // CHECK-NEXT: ry({{.*}}) [[q0]];
  "qir.Ry"(%q0, %c2) : (!qir.qubit, f64) -> ()
  // CHECK-NEXT: rz({{.*}}) [[q1]];
  "qir.Rz"(%q1, %c3) : (!qir.qubit, f64) -> ()
  // CHECK-NEXT: u2({{.*}},{{.*}}) [[q2]];
  "qir.U2"(%q2, %c1, %c2) : (!qir.qubit, f64, f64) -> ()
  // CHECK-NEXT: u1({{.*}}) [[q0]];
  "qir.U1"(%q0, %c3) : (!qir.qubit, f64) -> ()
  // CHECK-NEXT: s [[q1]];
  "qir.S"(%q1)      : (!qir.qubit) -> ()
  // CHECK-NEXT: sdg [[q2]];
  "qir.Sdg"(%q2)    : (!qir.qubit) -> ()
  // CHECK-NEXT: t [[q0]];
  "qir.T"(%q0)      : (!qir.qubit) -> ()
  // CHECK-NEXT: tdg [[q1]];
  "qir.Tdg"(%q1)    : (!qir.qubit) -> ()
  // Extended multi-qubit gates
  // CHECK-NEXT: cz [[q0]], [[q2]];
  "qir.Cz"(%q0, %q2)   : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-NEXT: crz({{.*}}) [[q1]], [[q0]];
  "qir.CRz"(%q1, %q0, %c3) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-NEXT: cry({{.*}}) [[q2]], [[q1]];
  "qir.CRy"(%q2, %q1, %c2) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-NEXT: ccx [[q0]], [[q1]], [[q2]];
  "qir.CCX"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
  // CHECK-NEXT: barrier [[q0]], [[q1]], [[q2]];
  "qir.barrier"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
  // CHECK-NEXT: swap [[q1]], [[q2]];
  "qir.swap"(%q1, %q2) : (!qir.qubit, !qir.qubit) -> ()
  // Measurement utilities
  // CHECK-NEXT: measure [[q2]] -> [[c1]][0];
  "qir.measure"(%q2, %r1) : (!qir.qubit, !qir.result) -> ()
  // CHECK-NOT: "qir.read_measurement"
  %mread = "qir.read_measurement"(%r1) : (!qir.result) -> i1
  // CHECK-NEXT: reset [[q2]];
  "qir.reset"(%q2) : (!qir.qubit) -> ()
}
