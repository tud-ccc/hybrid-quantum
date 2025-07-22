// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file| FileCheck %s

// CHECK-LABEL: func.func @check_convert_XOp(
// CHECK: ) -> !quantum.qubit<1> {
func.func @check_convert_XOp(%b : i1) -> (!qillr.qubit) {
  // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  // CHECK-NOT: "qillr.alloc"()
  %q0 = "qillr.alloc" () : () -> (!qillr.qubit)

  scf.if %b {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.X"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
  }
  "qillr.X" (%q0) : (!qillr.qubit) -> ()
  // CHECK-DAG: return %[[Q1]]
  func.return %q0 : !qillr.qubit
}

 // -----
