// RUN: quantum-opt %s -lift-qillr-to-quantum -split-input-file| FileCheck %s

// CHECK-LABEL: func.func @complete_example(
// CHECK: ) -> tensor<1xi1> {
func.func @complete_example() -> (tensor<1xi1>) {
  // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1> 
  // CHECK-NOT: "qillr.alloc"()
  %q0 = "qillr.alloc" () : () -> (!qillr.qubit)
  // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  %q1 = "qillr.alloc" () : () -> (!qillr.qubit)
  // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  %q2 = "qillr.alloc" () : () -> (!qillr.qubit)

  // CHECK-NOT: "qillr.ralloc"
  %r0 = "qillr.ralloc" () : () -> (!qillr.result)

  // CHECK-DAG: %[[cst:.+]] = arith.constant
  %const1 = arith.constant 0.34 : f64
  %const2 = arith.constant 0.78 : f64

  // CHECK-DAG: %[[Q3:.+]] = "quantum.H"(%[[Q0]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  "qillr.H" (%q0) : (!qillr.qubit) -> ()
  // CHECK-DAG: %[[Q4:.+]] = "quantum.Rx"(%[[Q3]], %[[cst]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  "qillr.Rx" (%q0, %const1) : (!qillr.qubit, f64) -> ()
  // CHECK-DAG: %[[Q5:.+]] = "quantum.Ry"(%[[Q4]], %[[cst2:.+]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  "qillr.Ry" (%q0, %const2) : (!qillr.qubit, f64) -> ()
  // CHECK-DAG: %[[Q6:.+]] = "quantum.Rz"(%[[Q5]], %[[cst]]) : (!quantum.qubit<1>, f64) -> !quantum.qubit<1>
  "qillr.Rz" (%q0, %const1) : (!qillr.qubit, f64) -> ()

  // CHECK-DAG: %[[Q7:.+]], %[[Q8:.+]] = "quantum.SWAP"(%[[Q6]], %[[Q1]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  "qillr.swap"(%q0, %q1) : (!qillr.qubit, !qillr.qubit) -> ()

  // CHECK-DAG: %[[Q9:.+]], %[[Q10:.+]] = "quantum.CZ"(%[[Q7]], %[[Q8]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  "qillr.Cz"(%q0, %q1) : (!qillr.qubit, !qillr.qubit) -> ()

  // CHECK-DAG: %[[Q11:.+]], %[[Q12:.+]], %[[Q13:.+]] = "quantum.CCX"(%[[Q9]], %[[Q10]], %[[Q2]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
  "qillr.CCX"(%q0, %q1, %q2) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()

  // CHECK-DAG: %[[M:.+]], %[[Q14:.+]] = "quantum.measure_single"(%[[Q11]]) : (!quantum.qubit<1>) -> (i1, !quantum.qubit<1>)
  "qillr.measure" (%q0, %r0) : (!qillr.qubit, !qillr.result) -> ()
  // CHECK-NOT: "qillr.read_measurement"
  %m = "qillr.read_measurement" (%r0) : (!qillr.result) -> i1

  // CHECK-DAG: %[[MT:.+]] = tensor.from_elements %[[M]] : tensor<1xi1>
  %mt = tensor.from_elements %m : tensor<1xi1>

  // CHECK-DAG: %[[Q15:.+]]:3 = "quantum.barrier"(%[[Q14]], %[[Q12]], %[[Q13]]) : (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
  "qillr.barrier"(%q0, %q1, %q2) : (!qillr.qubit, !qillr.qubit, !qillr.qubit) -> ()

  // CHECK-DAG: "quantum.deallocate"(%[[Q16]]#0) : (!quantum.qubit<1>) -> ()
  "qillr.reset" (%q0) : (!qillr.qubit) -> ()
  // CHECK-DAG: "quantum.deallocate"(%[[Q16]]#1) : (!quantum.qubit<1>) -> ()
  "qillr.reset" (%q1) : (!qillr.qubit) -> ()
  // CHECK-DAG: "quantum.deallocate"(%[[Q15]]#2) : (!quantum.qubit<1>) -> ()
  "qillr.reset" (%q2) : (!qillr.qubit) -> ()
  // CHECK-DAG: return %[[MT]]
  func.return %mt : tensor<1xi1>
}
