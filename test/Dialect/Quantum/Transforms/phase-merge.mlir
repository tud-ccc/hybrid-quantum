// RUN: quantum-opt --canonicalize %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @merge_phase_x_vars(
  // CHECK: %[[Q:.+]]: {{.*}}, %[[T1:.+]]: {{.*}}, %[[T2:.+]]: {{.*}})
  func.func @merge_phase_x_vars(%q : !quantum.qubit<1>, %theta1 : f64, %theta2 : f64) -> !quantum.qubit<1> {
    // CHECK-DAG: %[[P:.+]] = arith.addf %[[T1]], %[[T2]] : f64
    // CHECK-DAG: %[[Q1:.+]] = "quantum.Rx"(%[[Q]], %[[P]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    %qz = "quantum.Rx"(%q, %theta1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.Rx"
    %qzz = "quantum.Rx"(%qz, %theta2) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK: return %[[Q1]]
    return %qzz : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @merge_phase_z_vars(
  // CHECK: %[[Q:.+]]: {{.*}}, %[[T1:.+]]: {{.*}}, %[[T2:.+]]: {{.*}})
  func.func @merge_phase_z_vars(%q : !quantum.qubit<1>, %theta1 : f64, %theta2 : f64) -> !quantum.qubit<1> {
    // CHECK-DAG: %[[P:.+]] = arith.addf %[[T1]], %[[T2]] : f64
    // CHECK-DAG: %[[Q1:.+]] = "quantum.Rz"(%[[Q]], %[[P]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    %qz = "quantum.Rz"(%q, %theta1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.Rz"
    %qzz = "quantum.Rz"(%qz, %theta2) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK: return %[[Q1]]
    return %qzz : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @merge_phase_y_vars(
  // CHECK: %[[Q:.+]]: {{.*}}, %[[T1:.+]]: {{.*}}, %[[T2:.+]]: {{.*}})
  func.func @merge_phase_y_vars(%q : !quantum.qubit<1>, %theta1 : f64, %theta2 : f64) -> !quantum.qubit<1> {
    // CHECK-DAG: %[[P:.+]] = arith.addf %[[T1]], %[[T2]] : f64
    // CHECK-DAG: %[[Q1:.+]] = "quantum.Ry"(%[[Q]], %[[P]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    %qz = "quantum.Ry"(%q, %theta1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.Ry"
    %qzz = "quantum.Ry"(%qz, %theta2) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK: return %[[Q1]]
    return %qzz : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @merge_phase_y_const(
  // CHECK: %[[Q:.+]]: {{.*}})
  func.func @merge_phase_y_const(%q : !quantum.qubit<1>) -> !quantum.qubit<1> {
    // CHECK-DAG: %[[P:.+]] = arith.constant
    %theta1 = arith.constant 0.1 : f64
    // CHECK-NOT: arith.constant
    %theta2 = arith.constant 0.2 : f64
    // CHECK-DAG: %[[Q1:.+]] = "quantum.Ry"(%[[Q]], %[[P]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    %qz = "quantum.Ry"(%q, %theta1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.Ry"
    %qzz = "quantum.Ry"(%qz, %theta2) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
    // CHECK: return %[[Q1]]
    return %qzz : !quantum.qubit<1>
  }
}
