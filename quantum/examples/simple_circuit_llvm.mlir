module {
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__h__body(!llvm.ptr)
  llvm.func @__quantum__rt__result_allocate() -> !llvm.ptr
  llvm.func @__quantum__rt__qubit_allocate() -> !llvm.ptr
  llvm.func @main() {
    %0 = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
    %1 = llvm.call @__quantum__rt__result_allocate() : () -> !llvm.ptr
    llvm.call @__quantum__qis__h__body(%0) : (!llvm.ptr) -> ()
    llvm.call @__quantum__qis__mz__body(%0, %1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}

