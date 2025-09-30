// RUN: quantum-opt %s --debug --mlir-print-ir-after-all -control-flow-hoisting -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @hoist_binary_op(
// CHECK-SAME: %[[Q1:.+]]: {{.*}}, %[[Q2:.+]]: {{.*}}, %[[B:.+]]: {{.*}})
func.func @hoist_binary_op(%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>, %b : i1) -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q3:.+]], %[[Q4:.+]] = "quantum.CNOT"(%[[Q1]], %[[Q2]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    // CHECK-DAG: %[[PRED:.+]] = rvsdg.match(%[[B]] : i1) [#rvsdg.matchRule<0 -> 0>, #rvsdg.matchRule<1 -> 1>] -> <2>
    %predicate = rvsdg.match(%b : i1) [
        #rvsdg.matchRule<0 -> 0>,
        #rvsdg.matchRule<1 -> 1>
    ] -> !rvsdg.ctrl<2>
    // CHECK-DAG: %[[OUT2:.+]]:2 = rvsdg.gamma(%[[PRED]] : <2>) (%[[Q3]]: !quantum.qubit<1>, %[[Q4]]: !quantum.qubit<1>) : [
    %aout, %bout = rvsdg.gamma (%predicate : !rvsdg.ctrl<2>) (%q1 : !quantum.qubit<1>, %q2 : !quantum.qubit<1>):[
        // CHECK-NEXT: (%[[A0:.+]]: !quantum.qubit<1>, %[[B0:.+]]: !quantum.qubit<1>): {
        (%a0 : !quantum.qubit<1>, %b0: !quantum.qubit<1>): {
            // CHECK-NOT: "quantum.CNOT"
            %nota, %notb = "quantum.CNOT" (%a0, %b0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
            // CHECK-DAG: rvsdg.yield (%[[A0]]: !quantum.qubit<1>, %[[B0]]: !quantum.qubit<1>)
            rvsdg.yield (%nota : !quantum.qubit<1>, %notb : !quantum.qubit<1>)
        // CHECK: },
        },
        // CHECK-NEXT: (%[[A0:.+]]: !quantum.qubit<1>, %[[B0:.+]]: !quantum.qubit<1>): { 
        (%a0 : !quantum.qubit<1>, %b0: !quantum.qubit<1>): {
            // CHECK-NOT: "quantum.CNOT"
            %nota, %notb = "quantum.CNOT" (%a0, %b0) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
            // CHECK-DAG: rvsdg.yield (%[[A0]]: !quantum.qubit<1>, %[[B0]]: !quantum.qubit<1>)
            rvsdg.yield (%nota : !quantum.qubit<1>, %notb : !quantum.qubit<1>)
        }
    ] -> !quantum.qubit<1>, !quantum.qubit<1>

    // CHECK-DAG: return %[[OUT2]]#0, %[[OUT2]]#1
    return %aout, %bout : !quantum.qubit<1>, !quantum.qubit<1>
}

// -----
