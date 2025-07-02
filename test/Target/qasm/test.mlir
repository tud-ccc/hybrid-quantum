// RUN: quantum-translate --mlir-to-openqasm %s | FileCheck %s

module {
  // CHECK: OPENQASM 2.0;
  // CHECK-NEXT: include "qelib1.inc";
  // CHECK-DAG: qreg [[q0:reg.+]][1]; 
  %q0 = "qillr.alloc"() : () -> (!qillr.qubit)
  // CHECK-NEXT: qreg [[q1:reg.+]][1];
  %q1 = "qillr.alloc"() : () -> (!qillr.qubit)
  // CHECK-NEXT: qreg [[q2:reg.+]][1];
  %q2 = "qillr.alloc"() : () -> (!qillr.qubit)
  // CHECK-NEXT: creg [[c0:reg.+]][1];
  %r0 = "qillr.ralloc"() : () -> (!qillr.result)
  // CHECK-NEXT: creg [[c1:reg.+]][1];
  %r1 = "qillr.ralloc"() : () -> (!qillr.result)
  // Basic gates and operations
  // CHECK-NEXT: h [[q0]];
  "qillr.H"(%q0): (!qillr.qubit) -> ()
  // CHECK-NEXT: x [[q1]];
  "qillr.X"(%q1): (!qillr.qubit) -> ()
  // CHECK-NOT: arith.constant
  %c1 = arith.constant 0.1 : f64
  %c2 = arith.constant 0.2 : f64
  %c3 = arith.constant 0.3 : f64
  // CHECK-NEXT: u3({{.+}},{{.+}},{{.+}}) [[q2]];
  "qillr.U3"(%q2, %c1, %c2, %c3) : (!qillr.qubit, f64, f64, f64) -> ()
  // CHECK-NEXT: cx [[q0]], [[q1]];
  "qillr.CNOT"(%q0, %q1) : (!qillr.qubit, !qillr.qubit) -> ()
  // CHECK-NEXT: z [[q0]];
  "qillr.Z"(%q0)      : (!qillr.qubit) -> ()
  // CHECK-NEXT: y [[q1]];
  "qillr.Y"(%q1)      : (!qillr.qubit) -> ()
  // CHECK-NEXT: rx({{.*}}) [[q2]];
  "qillr.Rx"(%q2, %c1) : (!qillr.qubit, f64) -> ()
  // CHECK-NEXT: ry({{.*}}) [[q0]];
  "qillr.Ry"(%q0, %c2) : (!qillr.qubit, f64) -> ()
  // CHECK-NEXT: rz({{.*}}) [[q1]];
  "qillr.Rz"(%q1, %c3) : (!qillr.qubit, f64) -> ()
  // CHECK-NEXT: u2({{.*}},{{.*}}) [[q2]];
  "qillr.U2"(%q2, %c1, %c2) : (!qillr.qubit, f64, f64) -> ()
  // CHECK-NEXT: u1({{.*}}) [[q0]];
  "qillr.U1"(%q0, %c3) : (!qillr.qubit, f64) -> ()
  // CHECK-NEXT: s [[q1]];
  "qillr.S"(%q1)      : (!qillr.qubit) -> ()
  // CHECK-NEXT: sdg [[q2]];
  "qillr.Sdg"(%q2)    : (!qillr.qubit) -> ()
  // CHECK-NEXT: t [[q0]];
  "qillr.T"(%q0)      : (!qillr.qubit) -> ()
  // CHECK-NEXT: tdg [[q1]];
  "qillr.Tdg"(%q1)    : (!qillr.qubit) -> ()
  // Extended multi-qubit gates
  // CHECK-NEXT: cz [[q0]], [[q2]];
  "qillr.Cz"(%q0, %q2)   : (!qillr.qubit, !qillr.qubit) -> ()
  // CHECK-NEXT: crz({{.*}}) [[q1]], [[q0]];
  "qillr.CRz"(%q1, %q0, %c3) : (!qillr.qubit, !qillr.qubit, f64) -> ()
  // CHECK-NEXT: cry({{.*}}) [[q2]], [[q1]];
  "qillr.CRy"(%q2, %q1, %c2) : (!qillr.qubit, !qillr.qubit, f64) -> ()
  // CHECK-NEXT: ccx [[q0]], [[q1]], [[q2]];
  "qillr.CCX"(%q0, %q1, %q2) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
  // CHECK-NEXT: barrier [[q0]], [[q1]], [[q2]];
  "qillr.barrier"(%q0, %q1, %q2) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()
  // CHECK-NEXT: swap [[q1]], [[q2]];
  "qillr.swap"(%q1, %q2) : (!qillr.qubit, !qillr.qubit) -> ()
  // Measurement utilities
  // CHECK-NEXT: measure [[q2]] -> [[c1]][0];
  "qillr.measure"(%q2, %r1) : (!qillr.qubit, !qillr.result) -> ()
  // CHECK-NOT: "qillr.read_measurement"
  %mread = "qillr.read_measurement"(%r1) : (!qillr.result) -> i1
  // CHECK-NEXT: reset [[q2]];
  "qillr.reset"(%q2) : (!qillr.qubit) -> ()
}
