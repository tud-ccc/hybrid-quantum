// RUN: quantum-opt -convert-qir-to-llvm -convert-func-to-llvm \
// RUN:   -convert-arith-to-llvm -one-shot-bufferize="allow-unknown-ops" \
// RUN:   -finalize-memref-to-llvm %s -o - | FileCheck %s

//===----------------------------------------------------------------------===//
// Check runtime declarations: seed and initialization
//===----------------------------------------------------------------------===//
// CHECK-DAG: llvm.func @set_rng_seed(i64)
// CHECK-DAG: llvm.func @__quantum__rt__initialize(!llvm.ptr)

//===----------------------------------------------------------------------===//
// Check gate-related declarations
//===----------------------------------------------------------------------===//
// CHECK-DAG: llvm.func @__quantum__qis__x__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__y__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__z__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__rx__body(f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__ry__body(f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__rz__body(f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__u1__body(f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__u2__body(f64, f64, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__crz__body(f64, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__cry__body(f64, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__ccx__body(!llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__s__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__sdg__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__t__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__tdg__body(!llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__swap__body(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__quantum__qis__read_result__body(!llvm.ptr) -> i1
// CHECK-DAG: llvm.func @__quantum__qis__reset__body(!llvm.ptr)

func.func @main() -> (i1) {
  //===----------------------------------------------------------------------===//
  // Seed RNG
  //===----------------------------------------------------------------------===//
  %cseed = arith.constant 123 : i64
  // CHECK-DAG: %[[SEED:.+]] = llvm.mlir.constant(123 : i64) : i64
  // CHECK-DAG: llvm.call @set_rng_seed(%[[SEED]]) : (i64) -> ()
  "qir.seed"(%cseed) : (i64) -> ()

  //===----------------------------------------------------------------------===//
  // Initialize QIR runtime
  //===----------------------------------------------------------------------===//
  // CHECK-DAG: %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK-DAG: llvm.call @__quantum__rt__initialize(%[[NULL]]) : (!llvm.ptr) -> ()
  "qir.init"() : () -> ()

  //===----------------------------------------------------------------------===//
  // Allocate qubits & result registers
  //===----------------------------------------------------------------------===//
  %q0 = "qir.alloc"() : () -> (!qir.qubit)
  // CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[Q0PTR:.+]] = llvm.inttoptr %[[C0]] : i64 to !llvm.ptr

  %q1 = "qir.alloc"() : () -> (!qir.qubit)
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG: %[[Q1PTR:.+]] = llvm.inttoptr %[[C1]] : i64 to !llvm.ptr

  %r0 = "qir.ralloc"() : () -> (!qir.result)
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[RPTR:.+]] = llvm.inttoptr %[[C2]] : i64 to !llvm.ptr

  //===----------------------------------------------------------------------===//
  // Constants for rotations
  //===----------------------------------------------------------------------===//
  %c1 = arith.constant 0.34  : f64
  // CHECK-DAG: %[[F1:.+]] = llvm.mlir.constant(3.400000e-01 : f64) : f64
  %c2 = arith.constant 0.735 : f64
  // CHECK-DAG: %[[F2:.+]] = llvm.mlir.constant(7.350000e-01 : f64) : f64
  %c3 = arith.constant 0.23  : f64
  // CHECK-DAG: %[[F3:.+]] = llvm.mlir.constant(2.300000e-01 : f64) : f64

  //===----------------------------------------------------------------------===//
  // Primitive single-qubit gates
  //===----------------------------------------------------------------------===//

  "qir.X"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__x__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  "qir.Y"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__y__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  "qir.Z"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__z__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  "qir.H"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__h__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  //===----------------------------------------------------------------------===//
  // Rotation gates
  //===----------------------------------------------------------------------===//

  "qir.Rx"(%q0, %c1) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__rx__body(%[[F1]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()

  "qir.Ry"(%q0, %c2) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__ry__body(%[[F2]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()

  "qir.Rz"(%q0, %c3) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__rz__body(%[[F3]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()

  //===----------------------------------------------------------------------===//
  // Parameterized single-qubit gates
  //===----------------------------------------------------------------------===//

  "qir.U1"(%q0, %c1) : (!qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__u1__body(%[[F1]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()

  "qir.U2"(%q0, %c2, %c3) : (!qir.qubit, f64, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__u2__body(%[[F2]], %[[F3]], %[[Q0PTR]]) : (f64, f64, !llvm.ptr) -> ()

  "qir.U3"(%q0, %c1, %c2, %c3) : (!qir.qubit, f64, f64, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__rz__body(%[[F2]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__ry__body(%[[F1]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__rz__body(%[[F3]], %[[Q0PTR]]) : (f64, !llvm.ptr) -> ()

  //===----------------------------------------------------------------------===//
  // Controlled and multi-qubit gates
  //===----------------------------------------------------------------------===//

  "qir.CNOT"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__cnot__body(%[[Q0PTR]], %[[Q1PTR]]) : (!llvm.ptr, !llvm.ptr) -> ()

  "qir.CRz"(%q0, %q1, %c1) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__crz__body(%[[F1]], %[[Q0PTR]], %[[Q1PTR]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()

  "qir.CRy"(%q1, %q0, %c2) : (!qir.qubit, !qir.qubit, f64) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__cry__body(%[[F2]], %[[Q1PTR]], %[[Q0PTR]]) : (f64, !llvm.ptr, !llvm.ptr) -> ()

  "qir.CCX"(%q0, %q1, %q0) : (!qir.qubit, !qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__ccx__body(%[[Q0PTR]], %[[Q1PTR]], %[[Q0PTR]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

  "qir.S"(%q1) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__s__body(%[[Q1PTR]]) : (!llvm.ptr) -> ()
  "qir.Sdg"(%q1) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__sdg__body(%[[Q1PTR]]) : (!llvm.ptr) -> ()

  "qir.T"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__t__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()
  "qir.Tdg"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__tdg__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  "qir.swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__swap__body(%[[Q0PTR]], %[[Q1PTR]]) : (!llvm.ptr, !llvm.ptr) -> ()

  "qir.measure"(%q0, %r0) : (!qir.qubit, !qir.result) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__mz__body(%[[Q0PTR]], %[[RPTR]]) : (!llvm.ptr, !llvm.ptr) -> ()
  %mt = "qir.read_measurement"(%r0) : (!qir.result) -> i1
  // CHECK-DAG: llvm.call @__quantum__qis__read_result__body(%[[RPTR]]) : (!llvm.ptr) -> i1

  "qir.reset"(%q0) : (!qir.qubit) -> ()
  // CHECK-DAG: llvm.call @__quantum__qis__reset__body(%[[Q0PTR]]) : (!llvm.ptr) -> ()

  return %mt : i1
}
