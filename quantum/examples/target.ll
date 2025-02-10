; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Array = type opaque
%Result = type opaque
%Qubit = type opaque
%String = type opaque

declare %Array* @__quantum__rt__qubit_allocate_array(i64)
declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)
declare void @__quantum__qis__h__body(%Qubit*)
declare %Result* @__quantum__qis__m__body(%Qubit*)
declare void @__quantum__rt__qubit_release_array(%Array*)
declare %String* @__quantum__rt__result_to_string(%Result*)
declare void @__quantum__rt__message(%String*)

define void @__nvqpp__mlirgen__kernel() #0 {
  ; Allocate a qubit array
  %qubit_array = call %Array* @__quantum__rt__qubit_allocate_array(i64 1)
  
  ; Get pointer to the first qubit
  %qubit_ptr = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %qubit_array, i64 0)
  %qubit = bitcast i8* %qubit_ptr to %Qubit**
  %q = load %Qubit*, %Qubit** %qubit, align 8
  
  ; Apply Hadamard gate
  call void @__quantum__qis__h__body(%Qubit* %q)
  
  ; Measure the qubit
  %result = call %Result* @__quantum__qis__m__body(%Qubit* %q)
  
  ; Convert result to string for output
  %result_str = call %String* @__quantum__rt__result_to_string(%Result* %result)
  
  ; Output the result
  call void @__quantum__rt__message(%String* %result_str)
  
  ; Release the qubit array
  call void @__quantum__rt__qubit_release_array(%Array* %qubit_array)
  
  ret void
}

!llvm.module.flags = !{!0}
attributes #0 = { "entry_point"}
!0 = !{i32 2, !"Debug Info Version", i32 3}
