%Result = type opaque
%Qubit = type opaque

define void @ENTRYPOINT__main() #0 {
block_0:
  call void @__quantum__qis__h__body(ptr* inttoptr (i64 0 to ptr))
  call void @__quantum__rt__tuple_record_output(i64 0, i8* null)
  ret void
}

declare void @__quantum__qis__h__body(ptr)

declare void @__quantum__rt__tuple_record_output(i64, i8*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="base_profile" "required_num_qubits"="2" "required_num_results"="0" }
attributes #1 = { "irreversible" }

; module flags

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
