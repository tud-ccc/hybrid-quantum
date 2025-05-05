// RUN: quantum-opt --qir-optimise %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @decompose_u3_to_zyz() -> !qir.qubit {
  func.func @decompose_u3_to_zyz() -> !qir.qubit {

    // CHECK-DAG: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
    %q = "qir.alloc"() : () -> (!qir.qubit)

    // CHECK-DAG: %[[CST1:.+]] = arith.constant {{.*}} : f64
    // CHECK-DAG: %[[CST2:.+]] = arith.constant {{.*}} : f64
    // CHECK-DAG: %[[CST3:.+]] = arith.constant {{.*}} : f64
    %theta = arith.constant 1.5708 : f64
    %phi = arith.constant 0.7854 : f64
    %lambda = arith.constant 3.1415 : f64

    // CHECK-DAG: "qir.Rz"(%[[Q]], %[[CST2]]) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: "qir.Ry"(%[[Q]], %[[CST3]]) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: "qir.Rz"(%[[Q]], %[[CST1]]) : (!qir.qubit, f64) -> ()
    "qir.U3"(%q, %theta, %phi, %lambda) : (!qir.qubit, f64, f64, f64) -> ()

    // CHECK-DAG: return %[[Q]] : !qir.qubit
    return %q : !qir.qubit
  }
}
