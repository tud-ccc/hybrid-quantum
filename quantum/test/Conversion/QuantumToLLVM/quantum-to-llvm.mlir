// RUN: quantum-opt %s --convert-arith-to-llvm --convert-quantum-to-llvm --canonicalize | FileCheck %s

module {

    // CHECK-LABEL: func.func @single_qubit(
    func.func @single_qubit() -> (i1) {
        // CHECK-DAG: %[[Q:.+]] = arith.constant 0 : i32
        // CHECK-DAG: %[[R:.+]] = arith.constant 0 : i32
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: llvm.call @__quantum__qis__m__body(%[[Q]], %[[R]]) : (!Qubit*, !Result*) -> ()
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
        // CHECK-DAG: llvm.call @__quantum__rt__result_record_output(%Result* inttoptr (i64 0 to %Result*), i8* null)
        return %m : i1
    }

}