// RUN: quantum-opt %s | FileCheck %s

// CHECK: alloc
%q = "qir.alloc" () : () -> (!qir.qubit)
%r = "qir.ralloc" () : () -> (!qir.result)

"qir.H" (%q) : (!qir.qubit) -> ()
"qir.measure" (%q, %r) : (!qir.qubit, !qir.result) -> ()
