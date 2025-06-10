// RUN: quantum-opt %s --debug --mlir-print-ir-after-all -control-flow-hoisting | FileCheck %s

// CHECK-LABEL: func.func @hoist_op_from_if(
// CHECK-SAME: %[[Q1:.+]]: {{.*}}, %[[Q2:.+]]: {{.*}}, %[[B:.+]]: {{.*}})
func.func @hoist_op_from_if(%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>, %b : i1) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q3]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[OUT2]] = quantum.if %[[B]] ins(%[[B0]] = %[[Q2]]) -> (!quantum.qubit<1>) {
    %aout, %bout = quantum.if %b ins(%a0 = %q1, %b0 = %q2) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
        // CHECK-DAG: %[[B1]] = "quantum.X"(%[[B0]])
        %b1 = "quantum.X" (%b0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-NOT "quantum.H"
        %a2 = "quantum.H" (%a0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B2]] = "quantum.Y"(%[[B1]])
        %b2 = "quantum.Y" (%b1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: "quantum.yield"(%[[B2]])
        "quantum.yield" (%a2, %b2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    } else {
        // CHECK-NOT "quantum.H"
        %a2 = "quantum.H" (%a0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B1]] = "quantum.Z"(%[[B0]])
        %b1 = "quantum.Z" (%b0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[B2]] = "quantum.Y"(%[[B1]])
        %b2 = "quantum.Y" (%b1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: "quantum.yield"(%[[B2]])
        "quantum.yield" (%a2, %b2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    }
    // CHECK-DAG: return %[[Q3]], %[[OUT2]]
    return %aout, %bout : !quantum.qubit<1>, !quantum.qubit<1>
}
