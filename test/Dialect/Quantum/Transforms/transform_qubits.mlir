// RUN: quantum-opt %s --quantum-multi-qubit-legalize | FileCheck %s

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

    // CHECK-LABEL: func.func @chain_of_uses(
    func.func @chain_of_uses() -> () {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-DAG: %[[Q3:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q4:.+]] = "quantum.H"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
        %qx = "quantum.H" (%q) : (!quantum.qubit<2>) -> (!quantum.qubit<2>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q3]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%qx) : (!quantum.qubit<2>) -> ()
        return
    }

    // CHECK-LABEL: func.func @split_qubits(
    func.func @split_qubits() -> () {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q3:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<3>)
        // CHECK-NOT: "quantum.split"
        %q1, %q2 = "quantum.split" (%q) : (!quantum.qubit<3>) -> (!quantum.qubit<2>, !quantum.qubit<1>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q2]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q1) : (!quantum.qubit<2>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q3]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q2) : (!quantum.qubit<1>) -> ()
        return
    }

    // CHECK-LABEL: func.func @merge_qubits(
    func.func @merge_qubits() -> () {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q3:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q1 = "quantum.alloc" () : () -> (!quantum.qubit<1>)
        %q2 = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-NOT: "quantum.merge"
        %q3 = "quantum.merge" (%q1, %q2) : (!quantum.qubit<1>, !quantum.qubit<2>) -> (!quantum.qubit<3>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q2]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q3]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q3) : (!quantum.qubit<3>) -> ()
        return
    }

    // CHECK-LABEL: func.func @split_merge_qubits(
    func.func @split_merge_qubits() -> () {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-NOT: "quantum.split"
        %q1, %q2 = "quantum.split" (%q) : (!quantum.qubit<2>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
        // CHECK-NOT: "quantum.merge"
        %q3 = "quantum.merge" (%q1, %q2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q2]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%q3) : (!quantum.qubit<2>) -> ()
        return
    }

    // CHECK-LABEL: func.func @two_qubit_measurement(
    func.func @two_qubit_measurement() -> (tensor<2xi1>) {
        // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
        %q = "quantum.alloc" () : () -> (!quantum.qubit<2>)
        // CHECK-DAG: %[[M1:.+]], %[[Q3:.+]] = "quantum.measure"(%[[Q1]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
        // CHECK-DAG: %[[M2:.+]], %[[Q4:.+]] = "quantum.measure"(%[[Q2]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
        // CHECK-DAG: %[[M:.+]] = tensor.concat dim(0) %[[M1]], %[[M2]] : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1> 
        %m, %qm = "quantum.measure" (%q) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
        // CHECK-DAG: "quantum.deallocate"(%[[Q3]]) : (!quantum.qubit<1>) -> ()
        // CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
        "quantum.deallocate" (%qm) : (!quantum.qubit<2>) -> ()
        // CHECK-DAG: return %[[M]]
        return %m : tensor<2xi1>
    }

}