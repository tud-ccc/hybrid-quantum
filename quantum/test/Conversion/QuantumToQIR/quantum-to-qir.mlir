// RUN: quantum-opt %s --convert-quantum-to-qir --canonicalize | FileCheck %s

module {

    // CHECK-LABEL: func.func @single_qubit(
    func.func @single_qubit() -> (i1) {
        // CHECK-DAG: %[[Q:.+]] = qir.alloc () : !qir.qubit
        // CHECK-DAG: %[[R:.+]] = qir.ralloc () : !qir.result
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: qir.measure (%[[Q]], %[[R]])
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // 
        return %m : i1
    }

}