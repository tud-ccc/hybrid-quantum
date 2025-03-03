// RUN: quantum-opt -convert-func-to-llvm --convert-qir-to-llvm --mlir-print-ir-after-all %s -o - | FileCheck %s
  
  func.func @main() {
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    %q1 = "qir.alloc" () : () -> (!qir.qubit)
    %r0 = "qir.ralloc" () : () -> (!qir.result)
    %const1 = arith.constant 0.34 : f64
    %const2 = arith.constant 0.735 : f64
    "qir.H" (%q0) : (!qir.qubit) -> ()
    "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
    "qir.Rz" (%q0, %const2) : (!qir.qubit, f64) -> ()
    "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    return
  }

// CHECK: llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
// CHECK: llvm.func @__quantum__qis__reset__body(!llvm.ptr)
// CHECK: llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
// CHECK: llvm.func @__quantum__qis__rz__body(f64, !llvm.ptr)
// CHECK: llvm.func @__quantum__qis__h__body(!llvm.ptr)
// CHECK: llvm.func @main() {
// CHECK:   %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK:   %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// CHECK:   %2 = llvm.mlir.constant(1 : i64) : i64
// CHECK:   %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
// CHECK:   %4 = llvm.mlir.constant(0 : i64) : i64
// CHECK:   %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
// CHECK:   %6 = llvm.mlir.constant(3.400000e-01 : f64) : f64
// CHECK:   %7 = llvm.mlir.constant(7.350000e-01 : f64) : f64
// CHECK:   llvm.call @__quantum__qis__h__body(%1) : (!llvm.ptr) -> ()
// CHECK:   llvm.call @__quantum__qis__rz__body(%6, %1) : (f64, !llvm.ptr) -> ()
// CHECK:   llvm.call @__quantum__qis__rz__body(%7, %1) : (f64, !llvm.ptr) -> ()
// CHECK:   llvm.call @__quantum__qis__mz__body(%1, %5) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   llvm.call @__quantum__qis__reset__body(%1) : (!llvm.ptr) -> ()
// CHECK:   llvm.call @__quantum__rt__result_record_output(%5, %5) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:   llvm.return
// CHECK: }
