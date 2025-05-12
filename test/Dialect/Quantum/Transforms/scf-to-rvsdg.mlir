// RUN: quantum-opt %s --scf-to-rvsdg -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @if_to_rvsdg_gamma
// CHECK-SAME: (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
func.func @if_to_rvsdg_gamma(%b : i1, %q : !quantum.qubit<1>) -> (!quantum.qubit<1>) {
    // CHECK-DAG: %[[QR:.*]] = quantum.if %[[ARG0]] ins(%[[QIN:.*]] = %[[ARG1]]) ->
    %q1 = scf.if %b -> !quantum.qubit<1> {
        // CHECK-DAG: %[[QH:.*]] = "quantum.H"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        %qH = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[QH]] : !quantum.qubit<1>
        scf.yield %qH : !quantum.qubit<1>
    } else {
        // CHECK-DAG: %[[QX:.*]] = "quantum.X"(%[[QIN]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        %qX = "quantum.X" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        // CHECK-DAG: quantum.yield %[[QX]] : !quantum.qubit<1>
        scf.yield %qX : !quantum.qubit<1>
    }
    // CHECK-DAG: return %[[QR]]
    return %q1 : !quantum.qubit<1>
}