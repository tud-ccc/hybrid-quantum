// RUN: quantum-opt --hermitian-peephole --quantum-optimise %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @triple_h_cancel(
  func.func @triple_h_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-NOT "quantum.H"
    %q2 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-NOT "quantum.H"
    %q3 = "quantum.H" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q4:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.H" (%q3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: return %[[Q4]]
    return %q4 : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @interleaved_gates_no_cancel(
  func.func @interleaved_gates_no_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q2 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q3:.+]] = "quantum.X"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q3 = "quantum.X" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q4:.+]] = "quantum.H"(%[[Q3]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.H" (%q3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: return %[[Q4]] : !quantum.qubit<1>
    return %q4 : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @interleaved_gates_do_cancel(
  func.func @interleaved_gates_do_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q2 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-NOT "quantum.H"
    %q3 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q4:.+]] = "quantum.X"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.X" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-NOT "quantum.H"
    %q5 = "quantum.H" (%q3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: "quantum.deallocate"(%[[Q4]]) : (!quantum.qubit<1>) -> ()
    "quantum.deallocate" (%q4) : (!quantum.qubit<1>) -> ()
    // CHECK-DAG: return %[[Q1]] : !quantum.qubit<1>
    return %q5 : !quantum.qubit<1>
  }

  // Ensures different qubits are not affected.
  // CHECK-LABEL: func.func @multi_qubit_no_h_cancel(
  func.func @multi_qubit_no_h_cancel() -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q2 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q3:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q3 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
      // CHECK-DAG: %[[Q4:.+]] = "quantum.H"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.H" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %q3, %q4 : !quantum.qubit<1>, !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @single_h_no_cancel(
  func.func @single_h_no_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.H"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q2 = "quantum.H" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %q2 : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @single_x_no_cancel(
  func.func @single_x_no_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.X"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q2 = "quantum.X" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %q2 : !quantum.qubit<1>
  }

    // CHECK-LABEL: func.func @triple_x_cancel(
  func.func @triple_x_cancel() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    //CHECK-NOT "quantum.X"
    %q2 = "quantum.X" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    //CHECK-NOT "quantum.X"
    %q3 = "quantum.X" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q4:.+]] = "quantum.X"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.X" (%q3) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %q4 : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @multi_qubit_no_x_cancel(
  func.func @multi_qubit_no_x_cancel() -> (!quantum.qubit<1>, !quantum.qubit<1>) {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q2:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q2 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q3:.+]] = "quantum.X"(%[[Q1]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q3 = "quantum.X" (%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[Q4:.+]] = "quantum.X"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
    %q4 = "quantum.X" (%q2) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    return %q3, %q4 : !quantum.qubit<1>, !quantum.qubit<1>
  }


    // CHECK-LABEL: func.func @drop_z_before_measure(
  func.func @drop_z_before_measure() -> !quantum.qubit<1> {
    // CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
    %q1 = "quantum.alloc"() : () -> (!quantum.qubit<1>)
    // CHECK-NOT: "quantum.Z"
    %q2 = "quantum.Z"(%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    // CHECK-DAG: %[[MEAS:.+]], %[[QOUT:.+]] = "quantum.measure"(%[[Q1]]) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    %m, %qout = "quantum.measure"(%q2) : (!quantum.qubit<1>) -> (tensor<1xi1>, !quantum.qubit<1>)
    return %qout : !quantum.qubit<1>
  }

  // CHECK-LABEL: func.func @drop_z_before_measure_multiqubit(
  func.func @drop_z_before_measure_multiqubit() -> !quantum.qubit<3> {
  // CHECK-DAG: %[[Q:.*]] = "quantum.alloc"() : () -> !quantum.qubit<3>
  %q = "quantum.alloc"() : () -> (!quantum.qubit<3>)
  // CHECK-NOT: "quantum.Z"
  %qz = "quantum.Z"(%q) : (!quantum.qubit<3>) -> (!quantum.qubit<3>)
  // CHECK-DAG: %[[MEAS:.+]], %[[QOUT:.+]] = "quantum.measure"(%[[Q]]) : (!quantum.qubit<3>) -> (tensor<3xi1>, !quantum.qubit<3>)
  %m, %qout = "quantum.measure"(%qz) : (!quantum.qubit<3>) -> (tensor<3xi1>, !quantum.qubit<3>)
  return %qout : !quantum.qubit<3>
  }

  // CHECK-LABEL: func.func @no_drop_z_interleaved(
  func.func @no_drop_z_interleaved() -> !quantum.qubit<2> {
  // CHECK-DAG: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<2>
  %q = "quantum.alloc"() : () -> (!quantum.qubit<2>)
  // CHECK-DAG: %[[QZ:.+]] = "quantum.Z"(%[[Q]]) : (!quantum.qubit<2>) -> !quantum.qubit<2>
  %qz = "quantum.Z"(%q) : (!quantum.qubit<2>) -> (!quantum.qubit<2>)
  // CHECK-DAG: %[[QH:.+]] = "quantum.H"(%[[QZ]]) : (!quantum.qubit<2>) -> !quantum.qubit<2>
  %qh = "quantum.H"(%qz) : (!quantum.qubit<2>) -> (!quantum.qubit<2>)
  // CHECK-DAG: %[[M:.+]], %[[QOUT:.+]] = "quantum.measure"(%[[QH]]) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
  %m, %qout = "quantum.measure"(%qh) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
  return %qout : !quantum.qubit<2>
  }

  // CHECK-LABEL: func.func @no_drop_z_with_merge(
  func.func @no_drop_z_with_merge() -> !quantum.qubit<2> {
  // CHECK-DAG: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<3>
  %q = "quantum.alloc"() : () -> (!quantum.qubit<3>)
  // CHECK-DAG: %[[SPLIT:.+]]:3 = "quantum.split"(%[[Q]]) : (!quantum.qubit<3>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
  %q0, %q1, %q2 = "quantum.split"(%q) : (!quantum.qubit<3>) -> (!quantum.qubit<1>, !quantum.qubit<1>, !quantum.qubit<1>)
  // CHECK-DAG: %[[Q1Z:.+]] = "quantum.Z"(%[[SPLIT]]#1) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  %q1z = "quantum.Z"(%q1) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
  // CHECK-DAG: %[[QMERGED:.+]] = "quantum.merge"(%[[SPLIT]]#0, %[[Q1Z]]) : (!quantum.qubit<1>, !quantum.qubit<1>) -> !quantum.qubit<2>
  %qmerged = "quantum.merge"(%q0, %q1z) : (!quantum.qubit<1>, !quantum.qubit<1>) -> (!quantum.qubit<2>)
  // CHECK-DAG: %[[M:.+]], %[[QOUT:.+]] = "quantum.measure"(%[[QMERGED]]) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
  %m, %qout = "quantum.measure"(%qmerged) : (!quantum.qubit<2>) -> (tensor<2xi1>, !quantum.qubit<2>)
  // CHECK: return %[[QOUT]] : !quantum.qubit<2>
  return %qout : !quantum.qubit<2>
  }

}
