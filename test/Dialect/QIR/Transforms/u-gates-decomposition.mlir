// RUN: quantum-opt --qir-decompose-ugates %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @decompose_u3_to_zyz(
  // CHECK-SAME: %[[Q:.+]]:{{.*}}, %[[T:.+]]:{{.*}}, %[[P:.+]]:{{.*}}, %[[L:.+]]:{{.*}}) -> !qir.qubit {
  func.func @decompose_u3_to_zyz(%q :!qir.qubit, %theta : f64, %phi : f64, %lambda : f64) -> !qir.qubit {
    // CHECK-DAG: "qir.Rz"(%[[Q]], %[[P]]) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: "qir.Ry"(%[[Q]], %[[T]]) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: "qir.Rz"(%[[Q]], %[[L]]) : (!qir.qubit, f64) -> ()
    "qir.U3"(%q, %theta, %phi, %lambda) : (!qir.qubit, f64, f64, f64) -> ()

    // CHECK-DAG: return %[[Q]] : !qir.qubit
    return %q : !qir.qubit
  }
}
