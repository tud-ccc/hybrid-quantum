// RUN: quantum-opt %s -split-input-file -convert-scf-to-rvsdg | FileCheck %s

// CHECK-LABEL: func.func @if_to_rvsdg_gamma_quantum
// CHECK-SAME: (%[[B:.*]]: i1, %[[Q:.*]]: !quantum.qubit<1>)
func.func @if_to_rvsdg_gamma_quantum(%b : i1, %q : !quantum.qubit<1>) -> (!quantum.qubit<1>) {
  // CHECK-DAG: %[[PRED:.*]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
  // CHECK-DAG: %[[RES:.*]]:2 = rvsdg.gammaNode(%[[PRED]] : <2>) (%[[Q]]: !quantum.qubit<1>) : [
  // CHECK-NEXT:   (%[[QA:.*]]: !quantum.qubit<1>): {
  %q1 = scf.if %b -> !quantum.qubit<1> {
    // CHECK-DAG:     %[[QH:.*]] = "quantum.H"(%[[QA]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %qH = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG:     rvsdg.yield (%[[QH]]: !quantum.qubit<1>, %[[QA]]: !quantum.qubit<1>)
    scf.yield %qH : !quantum.qubit<1>
    // CHECK-NEXT:   },
  } else {
    // CHECK-NEXT:   (%[[QB:.*]]: !quantum.qubit<1>): {
    // CHECK-DAG:     %[[QX:.*]] = "quantum.X"(%[[QB]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %qX = "quantum.X" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG:     rvsdg.yield (%[[QX]]: !quantum.qubit<1>, %[[QB]]: !quantum.qubit<1>)
    scf.yield %qX : !quantum.qubit<1>
    // CHECK-NEXT:   }
  }
  // CHECK-NEXT: ] -> !quantum.qubit<1>
  // CHECK: return %[[RES]]#0
  return %q1 : !quantum.qubit<1>
}

// -----
