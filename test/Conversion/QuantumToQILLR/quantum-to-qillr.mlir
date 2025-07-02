// RUN: quantum-opt %s --convert-quantum-to-qillr -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @return_single_qubit(
// CHECK: ) -> tensor<1xi1> {
func.func @return_single_qubit() -> tensor<1xi1> {
    // CHECK-DAG: %[[Q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[R:.+]] = "qillr.ralloc"() : () -> !qillr.result
    // CHECK-DAG: "qillr.measure"(%[[Q]], %[[R]]) : (!qillr.qubit, !qillr.result) -> ()
    // CHECK-DAG: %[[M:.+]] = "qillr.read_measurement"(%[[R]]) : (!qillr.result) -> i1
    // CHECK-DAG: %[[MT:.+]] = tensor.from_elements %[[M]] : tensor<1xi1>
    %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    // CHECK-DAG: "qillr.reset"(%[[Q]]) : (!qillr.qubit) -> ()
    "quantum.deallocate" (%q_m) : (!quantum.qubit<1>) -> ()
    // CHECK-NEXT: return %[[MT]]
    func.return %m : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @return_single_measurement_result(
// CHECK: ) -> tensor<1xi1> {
func.func @return_single_measurement_result() -> (tensor<1xi1>) {
    // CHECK-DAG: %[[Q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[R:.+]] = "qillr.ralloc"() : () -> !qillr.result
    // CHECK-DAG: "qillr.measure"(%[[Q]], %[[R]]) : (!qillr.qubit, !qillr.result) -> ()
    // CHECK-DAG: %[[M:.+]] = "qillr.read_measurement"(%[[R]]) : (!qillr.result) -> i1
    // CHECK-DAG: %[[MT:.+]] = tensor.from_elements %[[M]] : tensor<1xi1>
    %m, %q_m = "quantum.measure" (%q) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    // CHECK-DAG: "qillr.reset"(%[[Q]]) : (!qillr.qubit) -> ()
    "quantum.deallocate" (%q_m) : (!quantum.qubit<1>) -> ()
    // CHECK-DAG: return %[[MT]]
    func.return %m : tensor<1xi1>
}

// -----

// CHECK-LABEL: func.func @convertHOp(
func.func @convertHOp() -> () {
    // CHECK-NEXT: %[[Q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
    %q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
    // CHECK-NEXT: "qillr.H"(%[[Q]]) : (!qillr.qubit) -> ()
    %q1 = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: "qillr.reset"(%[[Q]]) : (!qillr.qubit) -> ()
    "quantum.deallocate" (%q1) : (!quantum.qubit<1>) -> ()
    // CHECK-NEXT: return
    return
}

// -----

// CHECK-LABEL: func.func @convertSwap(
func.func @convertSwap() -> () {
    // CHECK-NEXT: %[[Q1:[0]+]] = "qillr.alloc"() : () -> !qillr.qubit
    // CHECK-NEXT: %[[Q2:[1]+]] = "qillr.alloc"() : () -> !qillr.qubit
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    %q2 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-NEXT: "qillr.swap"(%[[Q1]], %[[Q2]]) : (!qillr.qubit, !qillr.qubit) -> ()
    %q1_out, %q2_out = "quantum.SWAP"(%q1, %q2) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    return
}
