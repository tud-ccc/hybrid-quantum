// RUN: quantum-opt %s -cse -split-input-file | FileCheck %s
// --debug -mlir-print-ir-after-all

// CHECK-LABEL: @single_read_replaced
func.func public @single_read_replaced() -> (i1, i1) {
// CHECK: %[[q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
%0 = "qillr.alloc"() : () -> !qillr.qubit
// CHECK: %[[r:.+]] = "qillr.ralloc"() : () -> !qillr.result
%1 = "qillr.ralloc"() : () -> !qillr.result
// CHECK: "qillr.measure"(%[[q]], %[[r]]) : (!qillr.qubit, !qillr.result) -> ()
"qillr.measure"(%0, %1) : (!qillr.qubit, !qillr.result) -> ()
// CHECK: %[[m1:.+]] = "qillr.read_measurement"(%[[r]]) : (!qillr.result) -> i1
%3 = "qillr.read_measurement"(%1) : (!qillr.result) -> i1
%false = arith.constant false
// CHECK: %[[cmp:.+]] = arith.cmpi eq, %[[m1]], %false : i1
%4 = arith.cmpi eq, %3, %false : i1
// CHECK-NOT: "qillr.read_measurement"(%[[r]]) : (!qillr.result) -> i1
%5 = "qillr.read_measurement"(%1) : (!qillr.result) -> i1
// CHECK-NOT: arith.cmpi
%6 = arith.cmpi eq, %3, %false : i1
// CHECK: "qillr.reset"(%[[q]]) : (!qillr.qubit) -> ()
"qillr.reset"(%0) : (!qillr.qubit) -> ()
// CHECK: return %[[cmp]], %[[cmp]] : i1, i1
return %4, %6 : i1, i1
}


// -----

// CHECK-LABEL: @multiple_measure_read_replaced
func.func public @multiple_measure_read_replaced() -> (i1, i1, i1) {
// CHECK: %[[q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
%0 = "qillr.alloc"() : () -> !qillr.qubit
// CHECK: %[[r:.+]] = "qillr.ralloc"() : () -> !qillr.result
%1 = "qillr.ralloc"() : () -> !qillr.result
// CHECK: "qillr.measure"(%[[q]], %[[r]]) : (!qillr.qubit, !qillr.result) -> ()
"qillr.measure"(%0, %1) : (!qillr.qubit, !qillr.result) -> ()
// CHECK: %[[m1:.+]] = "qillr.read_measurement"(%[[r]]) : (!qillr.result) -> i1
%3 = "qillr.read_measurement"(%1) : (!qillr.result) -> i1
%false = arith.constant false
// CHECK: %[[ret1:.+]] = arith.cmpi ugt, %[[m1]], %false : i1
%4 = arith.cmpi ugt, %3, %false : i1
// CHECK-NOT: "qillr.read_measurement"(%[[r]]) : (!qillr.result) -> i1
%5 = "qillr.read_measurement"(%1) : (!qillr.result) -> i1
// CHECK: %[[ret2:.+]] = arith.cmpi eq, %[[m1]], %false : i1
%6 = arith.cmpi eq, %5, %false : i1    
"qillr.reset"(%0) : (!qillr.qubit) -> ()
// CHECK: "qillr.measure"(%[[q]], %[[r]]) : (!qillr.qubit, !qillr.result) -> ()
"qillr.measure"(%0, %1) : (!qillr.qubit, !qillr.result) -> ()
// CHECK: %[[m2:.+]] = "qillr.read_measurement"(%[[r]]) : (!qillr.result) -> i1
%7 = "qillr.read_measurement"(%1) : (!qillr.result) -> i1
// CHECK: return %[[ret1]], %[[ret2]], %[[m2]]
return %4, %6, %7 : i1, i1, i1
}


// -----
