// RUN: quantum-opt %s -inline | FileCheck %s

// CHECK-NOT: "qillr.gate"
"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "test"}> ({
  ^bb0(%q0: !qillr.qubit):
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
    "qillr.return"() : () -> ()
}) : () -> ()

// CHECK-LABEL: func.func @test_gate_inlining(
// CHECK: ) {
func.func @test_gate_inlining() {
  // CHECK-NEXT: %[[Q:.+]] = "qillr.alloc"() : () -> !qillr.qubit
  %q = "qillr.alloc"() : () -> (!qillr.qubit)
  // CHECK-NEXT: "qillr.X"(%[[Q]]) : (!qillr.qubit) -> ()
  "qillr.call"(%q) <{callee = @test}> : (!qillr.qubit) -> ()
  // CHECK-NEXT: "qillr.reset"(%[[Q]]) : (!qillr.qubit) -> ()
  "qillr.reset"(%q) : (!qillr.qubit) -> ()
  return
}
