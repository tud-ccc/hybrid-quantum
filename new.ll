module {
  llvm.func @__quantum__qis__read_result__body(!llvm.ptr) -> i1
  llvm.func @__quantum__rt__result_record_output(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__reset__body(!llvm.ptr)
  llvm.func @__quantum__qis__mz__body(!llvm.ptr, !llvm.ptr)
  llvm.func @__quantum__qis__rz__body(f64, !llvm.ptr)
  llvm.func @__quantum__qis__rx__body(f64, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @quantum_fn(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) -> i1 {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %12 = llvm.load %arg1 : !llvm.ptr -> f64
    %13 = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %14 = llvm.load %13 : !llvm.ptr -> f64
    llvm.call @__quantum__qis__rx__body(%12, %9) : (f64, !llvm.ptr) -> ()
    llvm.call @__quantum__qis__rz__body(%14, %9) : (f64, !llvm.ptr) -> ()
    llvm.call @__quantum__qis__mz__body(%9, %11) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @__quantum__qis__reset__body(%9) : (!llvm.ptr) -> ()
    llvm.call @__quantum__rt__result_record_output(%11, %11) : (!llvm.ptr, !llvm.ptr) -> ()
    %15 = llvm.call @__quantum__qis__read_result__body(%11) : (!llvm.ptr) -> i1
    llvm.return %15 : i1
  }
  llvm.func @main() -> i1 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(2.300000e+00 : f64) : f64
    %3 = llvm.mlir.constant(1.200000e+00 : f64) : f64
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[2] : (!llvm.ptr) -> !llvm.ptr, f64
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %4, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %5, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.store %3, %9 : f64, !llvm.ptr
    %17 = llvm.getelementptr %9[1] : (!llvm.ptr) -> !llvm.ptr, f64
    llvm.store %2, %17 : f64, !llvm.ptr
    %18 = llvm.call @quantum_fn(%9, %9, %13, %4, %5) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> i1
    llvm.return %18 : i1
  }
}

