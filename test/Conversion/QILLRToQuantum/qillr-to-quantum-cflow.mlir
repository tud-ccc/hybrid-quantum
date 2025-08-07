// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file | FileCheck %s

// CHECK-LABEL: inline_and_convert(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @inline_and_convert(%b : i1) {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: scf.if %[[B]] {
scf.if %b {
  // CHECK: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[Q3:.+]] = "quantum.X"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  // CHECK: "qqt.store"(%[[Q3]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
}
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

 // -----
