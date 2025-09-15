// RUN: quantum-opt %s -lift-qillr-to-quantum -hoist-load-store -eliminate-load-store -split-input-file | FileCheck %s
// --debug --mlir-print-ir-after-all

// CHECK-LABEL: if_local(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local(%b : i1) {
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q1]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>): { 
  // CHECK: %[[QX:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[QX]]: !quantum.qubit<1>)
  // CHECK },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_value_used(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_value_used(%b : i1) {
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q1]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>): { 
  // CHECK: %[[QX:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[QX]]: !quantum.qubit<1>)
  // CHECK },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK: %[[Q5:.+]] = "quantum.H"(%[[QOUT]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
"qillr.H" (%q1) : (!qillr.qubit) -> ()
// CHECK-DAG: "quantum.deallocate"(%[[Q5]]) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_chained(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_chained(%b : i1) {
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]] = rvsdg.gamma(%[[COND]] : <2>) (%[[Q1]]: !quantum.qubit<1>) : [
scf.if %b {
  // CHECK-NEXT: (%[[QIN:.+]]: !quantum.qubit<1>): { 
  // CHECK: %[[Q3:.+]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.X" (%q1) : (!qillr.qubit) -> ()
  // CHECK: %[[Q4:.+]] = "quantum.Z"(%[[Q3]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.Z" (%q1) : (!qillr.qubit) -> ()
  // CHECK: rvsdg.yield (%[[Q4]]: !quantum.qubit<1>)
  // CHECK: },
}
  // CHECK: (%[[QIN2:.+]]: !quantum.qubit<1>): {
  // CHECK: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
// CHECK: }
// CHECK: ] -> !quantum.qubit<1>
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_multiple_refs_exchanged(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_multiple_refs_exchanged(%b : i1) {
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q11:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q21:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]]:2  = rvsdg.gamma(%[[COND]] : <2>) (%[[Q21]]: !quantum.qubit<1>, %[[Q11]]: !quantum.qubit<1>) : [
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
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]#1) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]#0) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
return
}

// -----

// CHECK-LABEL: if_local_multiple_refs(
// CHECK-SAME: %[[B:.+]]: {{.*}})
func.func @if_local_multiple_refs(%b : i1) {
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q11:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NOT: "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q21:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
%q2 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK: %[[COND:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2> 
// CHECK: %[[QOUT:.+]]:2  = rvsdg.gamma(%[[COND]] : <2>) (%[[Q11]]: !quantum.qubit<1>, %[[Q21]]: !quantum.qubit<1>) : [
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
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]#0) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct" 
"qillr.reset" (%q1) : (!qillr.qubit) -> ()
// CHECK-DAG: "quantum.deallocate"(%[[QOUT]]#1) : (!quantum.qubit<1>) -> ()
// CHECK-NOT: "qqt.destruct"
"qillr.reset" (%q2) : (!qillr.qubit) -> ()
// CHECK: return
return
}

// -----

// CHECK-LABEL: multiple_if(
// CHECK-SAME: %[[B1:.+]]: {{.*}}, %[[B2:.+]]: {{.*}})
func.func @multiple_if(%b1 : i1, %b2 : i1) {
  // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  %1 = "qillr.alloc"() : () -> !qillr.qubit
  // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  %2 = "qillr.alloc"() : () -> !qillr.qubit
  // CHECK-DAG: %[[C1:.+]], %[[T1:.+]] = "quantum.CNOT"(%[[Q1]], %[[Q2]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  "qillr.CNOT"(%1, %2) : (!qillr.qubit, !qillr.qubit) -> ()
  // CHECK: %[[COND1:.+]] = rvsdg.match(%[[B1]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
  // CHECK: %[[QOUT:.+]]  = rvsdg.gamma(%[[COND1]] : <2>) (%[[C1]]: !quantum.qubit<1>) : [
  scf.if %b1 {
    // CHECK-NEXT: (%[[QIN1:.+]]: !quantum.qubit<1>): { 
    // CHECK-DAG: %[[cst1:.+]] = arith.constant 2.3561944901923448 : f64
    %cst1 = arith.constant 2.3561944901923448 : f64
    // CHECK: %[[QU1:.+]] = "quantum.U1"(%[[QIN1]], %[[cst1]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qillr.U1"(%1, %cst1) : (!qillr.qubit, f64) -> ()
    // CHECK: rvsdg.yield (%[[QU1]]: !quantum.qubit<1>)
    // CHECK: },
  }
    // CHECK-NEXT: (%[[QIN1:.+]]: !quantum.qubit<1>): { 
    // CHECK-NEXT: rvsdg.yield (%[[QIN1]]: !quantum.qubit<1>)
  // CHECK: }
  // CHECK: %[[COND2:.+]] = rvsdg.match(%[[B2]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
  // CHECK: %[[QOUT2:.+]]  = rvsdg.gamma(%[[COND2]] : <2>) (%[[QOUT]]: !quantum.qubit<1>) : [
  scf.if %b2 {
    // CHECK-NEXT: (%[[QIN2:.+]]: !quantum.qubit<1>): { 
    // CHECK: %[[cst2:.+]] = arith.constant 3.3561944901923448 : f64
    %cst2 = arith.constant 3.3561944901923448 : f64
    // CHECK: %[[QU2:.+]] = "quantum.U1"(%[[QIN2]], %[[cst2]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qillr.U1"(%1, %cst2) : (!qillr.qubit, f64) -> ()
    // CHECK: rvsdg.yield (%[[QU2]]: !quantum.qubit<1>)
    // CHECK: },
  }
    // CHECK-NEXT: (%[[QIN2:.+]]: !quantum.qubit<1>): { 
    // CHECK-NEXT: rvsdg.yield (%[[QIN2]]: !quantum.qubit<1>)
  // CHECK: "quantum.deallocate"(%[[QOUT2]])
  "qillr.reset" (%1) : (!qillr.qubit) -> ()
  // CHECK: "quantum.deallocate"(%[[T1]])
  "qillr.reset" (%2) : (!qillr.qubit) -> ()
  return
}
