; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Qubit = type opaque
%Result = type opaque
%Array = type opaque

@Qubit = external global %Qubit
@Result = external global %Result
@Array = external global %Array

declare ptr @__quantum__qis__mz__body(ptr, i32)

declare void @__quantum__qis__cnot(ptr)

declare void @__quantum__qis__h(ptr)

declare ptr @__quantum__rt__array_get_element_ptr_1d(ptr, i32)

declare ptr @__quantum__rt__qubit_allocate_array(i32)

define { i1, i1 } @entrypt() {
  %1 = call ptr @__quantum__rt__qubit_allocate_array(i32 2)
  %2 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %1, i32 0)
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %1, i32 1)
  %5 = load ptr, ptr %4, align 8
  call void @__quantum__qis__h(ptr %3)
  call void @__quantum__qis__cnot(ptr %3)
  %6 = call ptr @__quantum__qis__mz__body(ptr %3, i32 -1)
  %7 = load i1, ptr %6, align 1
  %8 = call ptr @__quantum__qis__mz__body(ptr %5, i32 -1)
  %9 = load i1, ptr %8, align 1
  %10 = insertvalue { i1, i1 } undef, i1 %7, 0
  %11 = insertvalue { i1, i1 } %10, i1 %9, 1
  ret { i1, i1 } %11
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
