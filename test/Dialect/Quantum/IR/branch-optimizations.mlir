// RUN: quantum-opt %s -control-flow-hoisting -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @hoist_unary_op(
// CHECK-SAME: %[[Q1:.+]]: {{.*}}, %[[Q2:.+]]: {{.*}}, %[[B:.+]]: {{.*}})
func.func @hoist_unary_op(%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>, %b : i1) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q3:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    // CHECK-DAG: %[[OUT2:.+]]:2 = quantum.if %[[B]] ins(%[[A0:.+]] = %[[Q3]], %[[B0:.+]] = %[[Q2]]) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    %aout, %bout = quantum.if %b ins(%a0 = %q1, %b0 = %q2) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        // CHECK-DAG: %[[B1:.+]] = "quantum.X"(%[[B0]])
        %b1 = "quantum.X" (%b0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-NOT "quantum.H"
        %a2 = "quantum.H" (%a0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B2:.+]] = "quantum.Y"(%[[B1]])
        %b2 = "quantum.Y" (%b1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[A0]], %[[B2]] : !quantum.qubit<1>, !quantum.qubit<1>
        "quantum.yield" (%a2, %b2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    } else {
        // CHECK-NOT "quantum.H"
        %a2 = "quantum.H" (%a0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B1:.+]] = "quantum.Z"(%[[B0]])
        %b1 = "quantum.Z" (%b0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B2:.+]] = "quantum.Y"(%[[B1]])
        %b2 = "quantum.Y" (%b1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[A0]], %[[B2]] : !quantum.qubit<1>, !quantum.qubit<1>
        "quantum.yield" (%a2, %b2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    }
    // CHECK-DAG: return %[[OUT2]]#0, %[[OUT2]]#1
    return %aout, %bout : !quantum.qubit<1>, !quantum.qubit<1>
}

// -----

// CHECK-LABEL: func.func @hoist_binary_op(
// CHECK-SAME: %[[Q1:.+]]: {{.*}}, %[[Q2:.+]]: {{.*}}, %[[B:.+]]: {{.*}})
func.func @hoist_binary_op(%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>, %b : i1) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q3:.+]], %[[Q4:.+]] = "quantum.CNOT"(%[[Q1]], %[[Q2]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    // CHECK-DAG: %[[OUT2:.+]]:2 = quantum.if %[[B]] ins(%[[A0:.+]] = %[[Q3]], %[[B0:.+]] = %[[Q4]]) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    %aout, %bout = quantum.if %b ins(%a0 = %q1, %b0 = %q2) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        // CHECK-NOT: "quantum.CNOT"
        %nota, %notb = "quantum.CNOT" (%a0, %b0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[A0]], %[[B0]] : !quantum.qubit<1>, !quantum.qubit<1>
        "quantum.yield" (%nota, %notb) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    } else {
        // CHECK-NOT: "quantum.CNOT"
        %nota, %notb = "quantum.CNOT" (%a0, %b0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[A0]], %[[B0]] : !quantum.qubit<1>, !quantum.qubit<1>
        "quantum.yield" (%nota, %notb) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    }
    // CHECK-DAG: return %[[OUT2]]#0, %[[OUT2]]#1
    return %aout, %bout : !quantum.qubit<1>, !quantum.qubit<1>
}

// -----

// CHECK-LABEL: func.func @hoist_rotation_op(
// CHECK-SAME: %[[Q1:.+]]: {{.*}}, %[[T1:.+]]: {{.*}}, %[[B:.+]]: {{.*}})
func.func @hoist_rotation_op(%q1 : !quantum.qubit<1>, %theta : f64, %b : i1) -> (!quantum.qubit<1>, f64) {
    // CHECK-DAG: %[[Q3:.+]] = "quantum.Rx"(%[[Q1]], %[[T1]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    // CHECK-DAG: %[[OUT2:.+]]:2 = quantum.if %[[B]] ins(%[[A0:.+]] = %[[Q3]], %[[T0:.+]] = %[[T1]]) -> (!quantum.qubit<1>, f64) {
    %aout, %tout = quantum.if %b ins(%a0 = %q1, %t0 = %theta) -> (!quantum.qubit<1>, f64) {
        // CHECK-NOT: "quantum.Rx"
        %rota = "quantum.Rx" (%a0, %t0) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
        // CHECK-DAG: quantum.yield %[[A0]], %[[T0]] : !quantum.qubit<1>, f64
        "quantum.yield" (%rota, %t0) : (!quantum.qubit<1>, f64) -> ()
    } else {
        // CHECK-NOT: "quantum.Rx"
        %rota = "quantum.Rx" (%a0, %t0) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
        // CHECK-DAG: quantum.yield %[[A0]], %[[T0]] : !quantum.qubit<1>, f64
        "quantum.yield" (%rota, %t0) : (!quantum.qubit<1>, f64) -> ()
    }
    // CHECK-DAG: return %[[OUT2]]#0, %[[OUT2]]#1
    return %aout, %tout : !quantum.qubit<1>, f64
}
