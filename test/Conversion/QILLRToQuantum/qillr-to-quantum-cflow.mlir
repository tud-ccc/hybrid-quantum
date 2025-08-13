// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file | FileCheck %s

// CHECK-LABEL: if_local(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local(%b : i1) {
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
// CHECK: }
}
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_value_used(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_value_used(%b : i1) {
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
// CHECK: }
}
// CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q5:.+]] = "quantum.H"(%[[Q4]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q5]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.H" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_chained(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_chained(%b : i1) {
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
  // CHECK: %[[Q4:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[Q5:.+]] = "quantum.Z"(%[[Q4]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  // CHECK: "qqt.store"(%[[Q5]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
  "qillr.Z" (%q1) : (!qillr.qubit) -> ()
  // CHECK: }
}
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----
