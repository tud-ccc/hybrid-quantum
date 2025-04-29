// RUN: quantum-opt -lift-qir-to-quantum | FileCheck %s
  
// CHECK-LABEL: func.func @main(
// CHECK: ) -> i1 {
func.func @main() -> (i1) {
  // CHECK-DAG: %[[Q0:.+]] = "quantum.alloc" () : () -> (!quantum.qubit<1>)
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc" () : () -> (!quantum.qubit<1>)
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  // CHECK-NOT "qir.ralloc"
  %r0 = "qir.ralloc" () : () -> (!qir.result)
  %const1 = arith.constant 0.34 : f64
  // CHECK-DAG: %[[Q2:.+]] = "quantum.H" (%[[Q0]]) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
  "qir.H" (%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: %[[Q3:.+]] = "quantum.Rz" (%[[Q2]], %const1) : (!quantum.qubit<1>, f64) -> (!quantum.qubit<1>)
  "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: %[[Q4:.+]], %[[Q5:.+]] = "quantum.SWAP" (%[[Q3]], %[[Q1]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<1>, !quantum.qubit<1>)
  "qir.swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: 
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  // CHECK-DAG: 
  %mt = "qir.read_measurement" (%r0) : (!qir.result) -> (tensor<1xi1>)
  // CHECK-DAG: 
  "qir.reset" (%q0) : (!qir.qubit) -> ()
  %i = "index.constant" () {value = 0 : index} : () -> (index)
  %m = "tensor.extract" (%mt, %i) : (tensor<1xi1>, index) -> (i1)
  // CHECK-DAG: llvm.return
  return %m : i1
}