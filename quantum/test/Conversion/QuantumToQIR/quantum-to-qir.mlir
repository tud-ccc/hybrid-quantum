// RUN: quantum-opt %s --convert-quantum-to-qir | FileCheck %s

module {

    // CHECK-LABEL: func.func @single_qubit(
    func.func @single_qubit() -> (i1) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: %[[R:.+]] = "qir.ralloc"() : () -> !qir.result
        // CHECK-NEXT: "qir.measure"(%[[Q]], %[[R]]) : (!qir.qubit, !qir.result) -> ()
        // CHECK-NEXT: %[[M:.+]] = "qir.read_measurement"(%[[R]]) : (!qir.result) -> i1
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-NEXT: return %[[M]]
        return %m : i1
    }

    // CHECK-LABEL: func.func @convertHOp(
    func.func @convertHOp() -> (i1) {
        // CHECK-NEXT: %[[Q:.+]] = "qir.alloc"() : () -> !qir.qubit
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-NEXT: "qir.H"(%[[Q]]) : (!qir.qubit) -> ()
        %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        return %m : i1
    }

}