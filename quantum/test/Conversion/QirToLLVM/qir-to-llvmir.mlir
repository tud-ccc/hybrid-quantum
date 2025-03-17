// RUN: quantum-opt -convert-qir-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -one-shot-bufferize="allow-unknown-ops" -finalize-memref-to-llvm %s -o - | FileCheck %s
  
// CHECK-DAG: llvm.func @__quantum__qis__reset__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__swap__body(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__read_result__body(!llvm.ptr) -> i1
// CHECK-DAG: llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__rz__body(f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__h__body(!llvm.ptr)

// CHECK-LABEL: llvm.func @main(
// CHECK: ) -> i1 {
func.func @main() -> (i1) {
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[C0PTR:.+]] = llvm.inttoptr %[[C0]] : i64 to !llvm.ptr
  %q0 = "qir.alloc" () : () -> (!qir.qubit)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG: %[[C1PTR:.+]] = llvm.inttoptr %[[C1]] : i64 to !llvm.ptr
  %q1 = "qir.alloc" () : () -> (!qir.qubit)
  // CHECK-DAG: %[[C0R:.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[C0RPTR:.+]] = llvm.inttoptr %[[C0R]] : i64 to !llvm.ptr
  %r0 = "qir.ralloc" () : () -> (!qir.result)
  // CHECK-DAG: %[[C034:.+]] = llvm.mlir.constant(3.400000e-01 : f64) : f64
  %const1 = arith.constant 0.34 : f64
  // CHECK-DAG: llvm.call @__quantum__qis__h__body(%[[C0PTR]]) : (!llvm.ptr) -> ()
  "qir.H" (%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__rz__body(%[[C034]], %[[C0PTR]]) : (f64, !llvm.ptr) -> ()
  "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__swap__body(%[[C0PTR]], %[[C1PTR]]) : (!llvm.ptr, !llvm.ptr) -> ()
  "qir.swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__mz__body(%[[C0PTR]], %[[C0RPTR]]) : (!llvm.ptr, !llvm.ptr) -> ()
  "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__read_result__body(%[[C0RPTR]]) : (!llvm.ptr) -> i1
  %mt = "qir.read_measurement" (%r0) : (!qir.result) -> (tensor<1xi1>)
  // CHECK-DAG: llvm.call @__quantum__qis__reset__body(%[[C0PTR]]) : (!llvm.ptr) -> ()
  "qir.reset" (%q0) : (!qir.qubit) -> ()
  %i = "index.constant" () {value = 0 : index} : () -> (index)
  %m = "tensor.extract" (%mt, %i) : (tensor<1xi1>, index) -> (i1)
  // CHECK-DAG: llvm.return
  return %m : i1
}