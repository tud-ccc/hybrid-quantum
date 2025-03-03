// RUN: quantum-opt -convert-func-to-llvm --convert-qir-to-llvm --mlir-print-ir-after-all %s -o - | quantum-translate --mlir-to-llvmir -o - --print-after-all | FileCheck %s
  
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

// CHECK: declare void @__quantum__rt__result_record_output(ptr, ptr)
// CHECK: declare void @__quantum__qis__reset__body(ptr)
// CHECK: declare void @__quantum__qis__mz__body(ptr, ptr)
// CHECK: declare void @__quantum__qis__rz__body(double, ptr)
// CHECK: declare void @__quantum__qis__h__body(ptr)
// CHECK: define void @main() {
// CHECK: call void @__quantum__qis__h__body(ptr null)
// CHECK: call void @__quantum__qis__rz__body(double 3.400000e-01, ptr null)
// CHECK: call void @__quantum__qis__rz__body(double 7.350000e-01, ptr null)
// CHECK: call void @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK: call void @__quantum__qis__reset__body(ptr null)
// CHECK: call void @__quantum__rt__result_record_output(ptr null, ptr null)
// CHECK: ret void
// CHECK: }
// CHECK: !llvm.module.flags = !{!0}
// CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
