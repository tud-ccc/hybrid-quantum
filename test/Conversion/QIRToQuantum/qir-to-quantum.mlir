// RUN: quantum-opt %s -lift-qir-to-quantum | FileCheck %s
// --debug --mlir-print-ir-after-all

module {
  // CHECK: "quantum.gate"() <{function_type = (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>), sym_name = "test"}> ({
  "qir.gate"() <{function_type = (!qir.qubit, !qir.qubit) -> (), sym_name = "test"}> ({
    // CHECK-NEXT: ^bb0(%[[QG1:.+]]: !quantum.qubit<1>, %[[QG2:.+]]: !quantum.qubit<1>):
    ^bb0(%arg1: !qir.qubit, %arg2: !qir.qubit):
    // CHECK-DAG: %[[QG3:.+]], %[[QG4:.+]] = "quantum.CNOT"(%[[QG1]], %[[QG2]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    "qir.CNOT"(%arg1, %arg2) : (!qir.qubit, !qir.qubit) -> ()
    // CHECK-DAG: "quantum.return"(%[[QG3]], %[[QG4]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> ()
    "qir.return"() : () -> ()
  }) : () -> ()

  // CHECK-LABEL: func.func @complete_example(
  // CHECK: ) -> tensor<1xi1> {
  func.func @complete_example() -> (tensor<1xi1>) {
    // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1> 
    // CHECK-NOT: "qir.alloc"()
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q2 = "qir.alloc" () : () -> (!qir.qubit)

    // CHECK-NOT: "qir.ralloc"
    %r0 = "qir.ralloc" () : () -> (!qir.result)

    // CHECK-DAG: %[[cst:.+]] = arith.constant
    %const1 = arith.constant 0.34 : f64
    %const2 = arith.constant 0.78 : f64

    // CHECK-DAG: %[[Q3:.+]] = "quantum.H"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qir.H" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: %[[Q4:.+]] = "quantum.Rx"(%[[Q3]], %[[cst]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qir.Rx" (%q0, %const1) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: %[[Q5:.+]] = "quantum.Ry"(%[[Q4]], %[[cst2:.+]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qir.Ry" (%q0, %const2) : (!qir.qubit, f64) -> ()
    // CHECK-DAG: %[[Q6:.+]] = "quantum.Rz"(%[[Q5]], %[[cst]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
    "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()

    // CHECK-DAG: %[[Q7:.+]], %[[Q8:.+]] = "quantum.SWAP"(%[[Q6]], %[[Q1]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    "qir.swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()

    // CHECK-DAG: %[[Q9:.+]], %[[Q10:.+]] = "quantum.CZ"(%[[Q7]], %[[Q8]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    "qir.Cz"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()

    // CHECK-DAG: %[[Q11:.+]], %[[Q12:.+]], %[[Q13:.+]] = "quantum.CCX"(%[[Q9]], %[[Q10]], %[[Q2]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
    "qir.CCX"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()

    // CHECK-DAG: %[[M:.+]], %[[Q14:.+]] = "quantum.measure_single"(%[[Q11]]) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
    "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    // CHECK-NOT: "qir.read_measurement"
    %m = "qir.read_measurement" (%r0) : (!qir.result) -> i1

    // CHECK-DAG: %[[MT:.+]] = tensor.from_elements %[[M]] : tensor<1xi1>
    %mt = tensor.from_elements %m : tensor<1xi1>

    // CHECK-DAG: %[[Q15:.+]]:3 = "quantum.barrier"(%[[Q14]], %[[Q12]], %[[Q13]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
    "qir.barrier"(%q0, %q1, %q2) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()

    // CHECK-DAG: %[[Q16:.+]]:2 = "quantum.call"(%[[Q15]]#0, %[[Q15]]#1) <{callee = @test}> : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
    "qir.call"(%q0, %q1) <{callee = @test}> : (!qir.qubit, !qir.qubit) -> ()

    // CHECK-DAG: "quantum.deallocate"(%[[Q16]]#0) : (!quantum.qubit<1>) -> ()
    "qir.reset" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: "quantum.deallocate"(%[[Q16]]#1) : (!quantum.qubit<1>) -> ()
    "qir.reset" (%q1) : (!qir.qubit) -> ()
    // CHECK-DAG: "quantum.deallocate"(%[[Q15]]#2) : (!quantum.qubit<1>) -> ()
    "qir.reset" (%q2) : (!qir.qubit) -> ()
    // CHECK-DAG: return %[[MT]]
    func.return %mt : tensor<1xi1>
  }

  // CHECK-LABEL: func.func @check_convert_XOp(
  // CHECK: ) -> !quantum.qubit<1> {
  func.func @check_convert_XOp() -> (!qir.qubit) {
    // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    // CHECK-NOT: "qir.alloc"()
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    // CHECK-DAG: %[[Q1:.+]] = "quantum.X"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    "qir.X" (%q0) : (!qir.qubit) -> ()
    // CHECK-DAG: return %[[Q1]]
    func.return %q0 : !qir.qubit
  }
}
