// RUN: quantum-opt %s -inline -lift-qillr-to-quantum -split-input-file | FileCheck %s

// CHECK-NOT: "qillr.gate"
"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "convert_xop"}> ({
  ^bb0(%q0: !qillr.qubit):
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
    "qillr.return"() : () -> ()
}) : () -> ()

// CHECK-LABEL: inline_and_convert
func.func @inline_and_convert() {
// CHECK-DAG: %[[Ref1:.+]] = "qqt.promote"() : () -> !qqt.ref
// CHECK-DAG: %[[Q1:.+]] = "quantum.alloc"() : () -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q1]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
%q1 = "qillr.alloc" () : () -> (!qillr.qubit)
// CHECK-NOT: "qillr.call"
// CHECK: %[[Q2:.+]] = "qqt.load"(%[[Ref1]]) : (!qqt.ref) -> !quantum.qubit<1>
// CHECK: %[[Q3:.+]] = "quantum.X"(%[[Q2]]) : (!quantum.qubit<1>) -> !quantum.qubit<1>
// CHECK: "qqt.store"(%[[Q3]], %[[Ref1]]) : (!quantum.qubit<1>, !qqt.ref) -> ()
"qillr.call"(%q1) <{callee = @convert_xop}> : (!qillr.qubit) -> ()
return
}

 // -----
