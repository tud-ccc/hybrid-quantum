// RUN: quantum-opt %s -lift-qillr-to-quantum -hoist-load-store -split-input-file | FileCheck %s

// CHECK-LABEL: if_local(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local(%b : i1) {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NEXT: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q2]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>): { 
  // CHECK-NOT: "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[QX:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  // CHECK-NOT: "qqt.store"(%[[QX]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[QX]]: !quantum.qubit<1>)
  // CHECK },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[QOUT]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
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
// CHECK-NEXT: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q2]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>): { 
  // CHECK-NOT: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[QX:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[QX]]: !quantum.qubit<1>)
  // CHECK },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[QOUT]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
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
// CHECK-NEXT: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q2]]: !quantum.qubit<1>, %[[Ref1]]: !qqt.ref) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>, %[[RefIN:.+]]: !qqt.ref): { 
  // CHECK-NOT: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[Q3:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  // CHECK: "qqt.store"(%[[Q3]], %[[RefIN]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: %[[Q4:.+]] = "qqt.load"(%[[RefIN]]) : (!qqt.ref) -> !quantum.qubit<1>
  // CHECK: %[[Q5:.+]] = "quantum.Z"(%[[Q4]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.Z" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[Q5]]: !quantum.qubit<1>)
  // CHECK: },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>, %[[RefIN2:.+]]: !qqt.ref): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[QOUT]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_multiple_refs_exchanged(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_multiple_refs_exchanged(%b : i1) {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q11:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q11]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q21:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q21]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Q8:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]]:2  = rvsdg.gamma(%[[COND]] : <2>) (%[[Q8]]: !quantum.qubit<1>, %[[Q2]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN2:.+]]: !quantum.qubit<1>, %[[QIN1:.+]]: !quantum.qubit<1>): { 
  // CHECK: %[[Q3:.+]] = "quantum.X"(%[[QIN2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q2) : (!qillr.qubit) -> ()
  // CHECK: %[[Q4:.+]] = "quantum.Z"(%[[QIN1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.Z" (%q1) : (!qillr.qubit) -> ()
  // CHECK-NEXT: rvsdg.yield (%[[Q3]]: !quantum.qubit<1>, %[[Q4]]: !quantum.qubit<1>)
  // CHECK-NEXT: },
}
  // CHECK-NEXT: (%[[QIN2]]: !quantum.qubit<1>, %[[QIN1]]: !quantum.qubit<1>): {
  // CHECK-NEXT: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>, %[[QIN1]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>, !quantum.qubit<1>
// CHECK-DAG: "qqt.store"(%[[QOUT]]#1, %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[QOUT]]#0, %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q7:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q7]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_multiple_refs(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_multiple_refs(%b : i1) {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q11:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q11]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-DAG: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: %[[Ref2:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q21:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK-DAG: "qqt.store"(%[[Q21]], %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NEXT: %[[Q8:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]]:2  = rvsdg.gamma(%[[COND]] : <2>) (%[[Q2]]: !quantum.qubit<1>, %[[Q8]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN1:.+]]: !quantum.qubit<1>, %[[QIN2:.+]]: !quantum.qubit<1>): { 
  // CHECK: %[[Q3:.+]] = "quantum.X"(%[[QIN1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: %[[Q4:.+]] = "quantum.Z"(%[[QIN2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.Z" (%q2) : (!qillr.qubit) -> ()
  // CHECK-NEXT: rvsdg.yield (%[[Q3]]: !quantum.qubit<1>, %[[Q4]]: !quantum.qubit<1>)
  // CHECK-NEXT: },
}
  // CHECK-NEXT: (%[[QIN1]]: !quantum.qubit<1>, %[[QIN2]]: !quantum.qubit<1>): {
  // CHECK-NEXT: rvsdg.yield (%[[QIN1]]: !quantum.qubit<1>, %[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>, !quantum.qubit<1>
// CHECK-DAG: "qqt.store"(%[[QOUT]]#0, %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK-DAG: "qqt.store"(%[[QOUT]]#1, %[[Ref2]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
// CHECK: %[[Q6:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref1]]) : (!qqt.ref) -> () 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK: %[[Q7:.+]] = "qqt.load"(%[[Ref2]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[Q7]]) : (!quantum.qubit<1>) -> ()
// CHECK-DAG: "qqt.destruct"(%[[Ref2]]) : (!qqt.ref) -> () 
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
return
}

// -----
