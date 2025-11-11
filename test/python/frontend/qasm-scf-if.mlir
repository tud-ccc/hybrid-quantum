// RUN: %PYTHON qasm-import -t QILLR -i %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT: func.func public @qasm_main() {
// CHECK-DAG: %[[q0:.+]] = "qillr.alloc"() : () -> !qillr.qubit
// CHECK-DAG: %[[c0:.+]] = "qillr.ralloc"() : () -> !qillr.result
// CHECK-DAG: "qillr.measure"(%[[q0]], %[[c0]]) : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: %[[m1:.+]] = "qillr.read_measurement"(%[[c0]]) : (!qillr.result) -> i1
// CHECK-DAG: %[[false:.+]] = arith.constant false
// CHECK-DAG: %[[b:.+]] = arith.cmpi eq, %[[m1]], %[[false]] : i1
// CHECK-DAG: scf.if %[[b]] {
// CHECK-DAG: "qillr.X"(%[[q0]]) : (!qillr.qubit) -> ()
// CHECK-NEXT: }
// CHECK-DAG: %[[false1:.+]] = arith.constant false
// CHECK-DAG: %[[b1:.+]] = arith.cmpi eq, %[[m1]], %[[false1]] : i1
// CHECK-DAG: scf.if %[[b1]] {
// CHECK-DAG: "qillr.X"(%[[q0]]) : (!qillr.qubit) -> ()
// CHECK-NEXT: }
// CHECK-DAG: "qillr.measure"(%[[q0]], %[[c0]]) : (!qillr.qubit, !qillr.result) -> ()
// CHECK-DAG: %[[m2:.+]] = "qillr.read_measurement"(%[[c0]]) : (!qillr.result) -> i1
// CHECK-DAG: "qillr.reset"(%[[q0]]) : (!qillr.qubit) -> ()
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }

OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
if(c==0) x q[0];
if(c==0) x q[0];
measure q[0] -> c[0];
reset q[0];
