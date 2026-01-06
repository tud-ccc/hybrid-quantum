// RUN: quantum-opt %s --split-input-file -inline | FileCheck %s

// CHECK-NOT: "quantum.gate"
"quantum.gate"() <{function_type = (!quantum.qubit<1>) -> (!quantum.qubit<1>), sym_name = "test"}> ({
  ^bb0(%q0: !quantum.qubit<1>):
    %q1 = "quantum.X" (%q0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
    "quantum.return"(%q1) : (!quantum.qubit<1>) -> ()
}) : () -> ()

// CHECK-LABEL: func.func @test_gate_inlining(
// CHECK: ) {
func.func @test_gate_inlining() {
  // CHECK-NEXT: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
  %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
  // CHECK-NEXT: %[[Q1:.+]] = "quantum.X"(%[[Q]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
  %q1 = "quantum.call"(%q) <{callee = @test}> : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
  // CHECK-NEXT: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
  "quantum.deallocate"(%q1) : (!quantum.qubit<1>) -> ()
  return
}

//-------

module {
  qpu.module @qpu {

    // CHECK-NOT: "quantum.gate"
    "quantum.gate"() <{function_type = (!quantum.qubit<1>) -> (!quantum.qubit<1>), sym_name = "test_gate"}> ({
      ^bb0(%q0: !quantum.qubit<1>):
        %q1 = "quantum.X" (%q0) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
        "quantum.return"(%q1) : (!quantum.qubit<1>) -> ()
    }) : () -> ()

    // CHECK: "qpu.circuit"() {{.*}} sym_name = "test_circuit"}
    "qpu.circuit"() <{function_type = () -> (), sym_name = "test_circuit"}>({
      // CHECK-NEXT: %[[Q:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
      %q = "quantum.alloc"() : () -> (!quantum.qubit<1>)
      // CHECK-NEXT: %[[Q1:.+]] = "quantum.X"(%[[Q]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
      %q1 = "quantum.call"(%q) <{callee = @test_gate}> : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
      // CHECK-NEXT: "quantum.deallocate"(%[[Q1]]) : (!quantum.qubit<1>) -> ()
      "quantum.deallocate"(%q1) : (!quantum.qubit<1>) -> ()
      "qpu.return"() : () -> ()
    }) : () -> ()
  }
}
