// RUN: quantum-opt --quantum-multi-qubit-legalize --quantum-optimise %s | FileCheck %s  
  
// CHECK-LABEL: func.func @no_drop_z_with_merge(
func.func @no_drop_z_with_merge() {
    // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q = "quantum.alloc"() : () -> (!quantum.qubit<3>)
    // CHECK-NOT: "quantum.split"
    %q0, %q1, %q2 = "quantum.split"(%q) : (!quantum.qubit<3>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
    // CHECK-NOT: "quantum.Z"
    %q1z = "quantum.Z"(%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.merge"
    %qmerged = "quantum.merge"(%q0, %q1z) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)
    // CHECK-DAG: %[[M0:.+]], %[[Q0M:.+]] = "quantum.measure"(%[[Q0]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    // CHECK-DAG: %[[M1:.+]], %[[Q1M:.+]] = "quantum.measure"(%[[Q1]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    %m, %qout = "quantum.measure"(%qmerged) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
    // CHECK: "quantum.deallocate"(%[[Q0M]]) : (!quantum.qubit<1>) -> ()
    // CHECK: "quantum.deallocate"(%[[Q1M]]) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%qout) : (!quantum.qubit<2>) -> ()
    // CHECK: "quantum.deallocate"(%[[Q2]]) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate"(%q2) : (!quantum.qubit<1>) -> ()
    return
}
