// RUN: quantum-opt %s --debug --quantum-multi-qubit-legalize | FileCheck %s
// --debug-only=dialect-conversion

module {
    
    // CHECK-LABEL: func.func @single_qubit_alloc(
    func.func @single_qubit_alloc() -> () {
        // CHECK-DAG: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q) : (!quantum.qubit<1>) -> ()
        return
    }

    // CHECK-LABEL: func.func @two_qubit_alloc(
    func.func @two_qubit_alloc() -> () {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q2]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q) : (!quantum.qubit<2>) -> ()
        return
    }
}