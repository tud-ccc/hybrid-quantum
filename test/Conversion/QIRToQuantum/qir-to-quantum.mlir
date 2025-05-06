// RUN: quantum-opt %s -lift-qir-to-quantum | FileCheck %s

module {

  // CHECK-LABEL: func.func @complete_example(
  // CHECK: ) -> tensor<1xi1> {
  func.func @complete_example() -> (tensor<1xi1>) {
    // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1> 
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-NOT "qir.ralloc"
    %r0 = "qir.ralloc" () : () -> (!qir.result)
    // CHECK-DAG: %[[cst:.+]] = arith.constant 
    %const1 = arith.constant 0.34 : f64
    // CHECK-DAG: %[[Q2:.+]] = "quantum.H"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qir.H" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: %[[Q3:.+]] = "quantum.Rz"(%[[Q2]], %[[cst]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: %[[Q4:.+]], %[[Q5:.+]] = "quantum.SWAP"(%[[Q3]], %[[Q1]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    "qir.swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
    // CHECK-DAG: %[[M:.+]], %[[Q6:.+]] = "quantum.measure"(%[[Q4]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    // CHECK-NOT: "qir.read_measurement"
    %mt = "qir.read_measurement" (%r0) : (!qir.result) -> (tensor<1xi1>)
    // CHECK-DAG: "quantum.deallocate"(%[[Q6]]) : (!quantum.qubit<1>) -> () 
    "qir.reset" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: "quantum.deallocate"(%[[Q5]]) : (!quantum.qubit<1>) -> () 
    "qir.reset" (%q1) : (!qir.qubit) -> ()
    // CHECK-DAG: return %[[M]]
    func.return %mt : tensor<1xi1>
  }

  // CHECK-LABEL: func.func @check_convert_XOp(
  // CHECK: ) -> !quantum.qubit<1> {
  func.func @check_convert_XOp() -> (!qir.qubit) {
    // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1> 
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qir.X" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: return %[[Q1]]
    func.return %q0 : !qir.qubit
  }
}