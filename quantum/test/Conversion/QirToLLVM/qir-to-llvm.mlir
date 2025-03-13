// RUN: quantum-opt -convert-func-to-llvm --convert-qir-to-llvm --mlir-print-ir-after-all %s -o - | quantum-translate --mlir-to-llvmir -o - --print-after-all | FileCheck %s
//--debug-only=dialect-conversion

module {
  func.func @main() {
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    %q1 = "qir.alloc" () : () -> (!qir.qubit)
    %r0 = "qir.ralloc" () : () -> (!qir.result)
    %const1 = arith.constant 0.34 : f64
    %const2 = arith.constant 0.735 : f64
    "qir.H" (%q0) : (!qir.qubit) -> ()
    "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
    "qir.Swap"(%q0, %q1) : (!qir.qubit, !qir.qubit) -> ()
    "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    return
  }
}

// CHECK: declare void @__quantum__rt__result_record_output(ptr, ptr)
// CHECK: declare void @__quantum__qis__reset__body(ptr)
// CHECK: declare void @__quantum__qis__mz__body(ptr, ptr)
// CHECK: declare void @__quantum__qis__swap__body(ptr, ptr)
// CHECK: declare void @__quantum__qis__rz__body(double, ptr)
// CHECK: declare void @__quantum__qis__h__body(ptr)

// CHECK: define void @main() {
// CHECK:   call void @__quantum__qis__h__body(ptr null)
// CHECK:   call void @__quantum__qis__rz__body(double 3.400000e-01, ptr null)
// CHECK:   call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 1 to ptr))
// CHECK:   call void @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK:   call void @__quantum__qis__reset__body(ptr null)
// CHECK:   call void @__quantum__rt__result_record_output(ptr null, ptr null)
// CHECK:   ret void
// CHECK: }
