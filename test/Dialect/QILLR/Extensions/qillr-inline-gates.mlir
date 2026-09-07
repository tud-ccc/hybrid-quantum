// RUN: quantum-opt %s --split-input-file -inline | FileCheck %s

qpu.module @test {

// CHECK-NOT: "qillr.gate"
"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "test"}> ({
  ^bb0(%q0: !qillr.qubit):
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
    "qillr.return"() : () -> ()
}) : () -> ()

// CHECK-LABEL: func.func @test_gate_inlining(
// CHECK: ) {
func.func @test_gate_inlining() {
  // CHECK-NEXT: %[[Q:.+]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
  %q = "qillr.alloc"() <{size = 1 : i64}> : () -> (!qillr.qubit)
  // CHECK-NEXT: "qillr.X"(%[[Q]]) <{index = []}> : (!qillr.qubit) -> ()
  "qillr.call"(%q) <{callee = @test}> : (!qillr.qubit) -> ()
  // CHECK-NEXT: "qillr.reset"(%[[Q]]) <{index = []}> : (!qillr.qubit) -> ()
  "qillr.reset"(%q) : (!qillr.qubit) -> ()
  return
}

}



//-------

qpu.module @qpu {

// CHECK-NOT: "qillr.gate"
"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "test_gate"}> ({
  ^bb0(%q0: !qillr.qubit):
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
    "qillr.return"() : () -> ()
}) : () -> ()

// CHECK: "qpu.circuit"() {{.*}} sym_name = "test_circuit"}
"qpu.circuit"() <{function_type = () -> (), sym_name = "test_circuit"}>({
  // CHECK-NEXT: %[[Q:.+]] = "qillr.alloc"() <{size = 1 : i64}> : () -> !qillr.qubit
  %q = "qillr.alloc"() <{size = 1 : i64}> : () -> (!qillr.qubit)
  // CHECK-NEXT: "qillr.X"(%[[Q]]) <{index = []}> : (!qillr.qubit) -> ()
  "qillr.call"(%q) <{callee = @test_gate}> : (!qillr.qubit) -> ()
  // CHECK-NEXT: "qillr.reset"(%[[Q]]) <{index = []}> : (!qillr.qubit) -> ()
  "qillr.reset"(%q) : (!qillr.qubit) -> ()
  "qpu.return"() : () -> ()
}) : () -> ()
}
