// RUN: quantum-opt --qillr-decompose-ugates %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @decompose_u3_to_zyz(
  // CHECK-SAME: %[[Q:.+]]:{{.*}}, %[[T:.+]]:{{.*}}, %[[P:.+]]:{{.*}}, %[[L:.+]]:{{.*}}) -> !qillr.qubit {
  func.func @decompose_u3_to_zyz(%q :!qillr.qubit, %theta : f64, %phi : f64, %lambda : f64) -> !qillr.qubit {
    // CHECK-DAG: "qillr.Rz"(%[[Q]], %[[P]]) : (!qillr.qubit, f64) -> ()
    // CHECK-DAG: "qillr.Ry"(%[[Q]], %[[T]]) : (!qillr.qubit, f64) -> ()
    // CHECK-DAG: "qillr.Rz"(%[[Q]], %[[L]]) : (!qillr.qubit, f64) -> ()
    "qillr.U3"(%q, %theta, %phi, %lambda) : (!qillr.qubit, f64, f64, f64) -> ()

    // CHECK-DAG: return %[[Q]] : !qillr.qubit
    return %q : !qillr.qubit
  }
}
