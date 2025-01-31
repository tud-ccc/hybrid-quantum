// RUN: quantum-opt %s --convert-arith-to-llvm --convert-quantum-to-llvm --canonicalize | FileCheck %s

module {

    // CHECK-LABEL: func.func @single_qubit(
    func.func @single_qubit() -> (f32) {
        %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        // CHECK-DAG: %[[M:.+]] = llvm.call ("__quantum__qis__mz__body") : 
        %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (f32, !quantum.qubit<1>)
        // CHECK: return %[[M]]
        return %m : f32
    }

}