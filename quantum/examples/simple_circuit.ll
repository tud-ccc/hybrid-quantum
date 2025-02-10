; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @__quantum__qis__mz__body(ptr, ptr)

declare void @__quantum__qis__h__body(ptr)

declare ptr @__quantum__rt__result_allocate()

declare ptr @__quantum__rt__qubit_allocate()

define void @main() {
  %1 = call ptr @__quantum__rt__qubit_allocate()
  %2 = call ptr @__quantum__rt__result_allocate()
  call void @__quantum__qis__h__body(ptr %1)
  call void @__quantum__qis__mz__body(ptr %1, ptr %2)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
