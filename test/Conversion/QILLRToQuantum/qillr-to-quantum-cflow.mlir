// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file| FileCheck %s

// CHECK: "quantum.gate"() <{function_type = (!quantum.qubit<1>) -> !quantum.qubit<1>, sym_name = "check_convert_XOp"}> ({
"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "check_convert_XOp"}> ({
  // CHECK-NEXT: ^bb0(%[[Q0:.+]]: !quantum.qubit<1>, %[[B:.+]]: i1):
  ^bb0(%q0: !qillr.qubit):

  %b = arith.constant true

  scf.if %b {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.X"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
  }
  // CHECK-DAG: "quantum.return"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
  "qillr.return"() : () -> ()
}) : () -> ()

 // -----
