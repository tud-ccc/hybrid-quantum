// RUN: quantum-opt %s -inline -lift-qillr-to-quantum -split-input-file | FileCheck %s

"qillr.gate"() <{function_type = (!qillr.qubit) -> (), sym_name = "convert_xop"}> ({
  ^bb0(%q0: !qillr.qubit):
    "qillr.X" (%q0) : (!qillr.qubit) -> ()
    "qillr.return"() : () -> ()
}) : () -> ()

func.func @inline_and_convert() {
    %q = "qillr.alloc"() : () -> !qillr.qubit
    "qillr.call"(%q) <{callee = @convert_xop}> : (!qillr.qubit) -> ()
    return
}

 // -----
