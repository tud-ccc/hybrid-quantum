// RUN: quantum-opt %s -split-input-file -convert-scf-to-rvsdg | FileCheck %s

// CHECK-LABEL: func.func @if_to_rvsdg_gamma_qillr(
// CHECK-SAME: %[[B:.*]]: i1, %[[Q:.*]]: !qillr.qubit) -> !qillr.qubit
func.func @if_to_rvsdg_gamma_qillr(%b : i1, %q : !qillr.qubit) -> (!qillr.qubit) {
  // CHECK-DAG: %[[PRED:.*]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<1 -> 0>, #rvsdg.matchRule<0 -> 1>] -> <2>
  // CHECK-DAG: %[[QR:.*]] = rvsdg.gammaNode(%[[PRED]] : <2>) (%[[Q]]: !qillr.qubit) : [
  // CHECK-NEXT:   (%[[Q1:.*]]: !qillr.qubit): {
  scf.if %b {
    // CHECK:         "qillr.H"(%[[Q1]]) : (!qillr.qubit) -> ()
    "qillr.H" (%q) : (!qillr.qubit) -> ()
    // CHECK:         rvsdg.yield (%[[Q1]]: !qillr.qubit)
    // CHECK-NEXT:   },
  }
  // CHECK-NEXT:   (%[[Q2:.*]]: !qillr.qubit): {
  // CHECK-NEXT:     rvsdg.yield (%[[Q2]]: !qillr.qubit)
  // CHECK-NEXT:   }
  // CHECK-NEXT: ] -> !qillr.qubit
  // CHECK: return %[[QR]]
  return %q : !qillr.qubit
}

// -----
