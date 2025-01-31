// RUN: quantum-opt %s

//===----------------------------------------------------------------------===//
// Cannot reuse qubit
//===----------------------------------------------------------------------===//

%q = "quantum.alloc" () : () -> (!quantum.qubit<1>)
%q0_H = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)
// expected-error@below {{'quantum.H' op quantum operand '!quantum.qubit<1>' has already been used}}
%q0_X = "quantum.H" (%q) : (!quantum.qubit<1>) -> (!quantum.qubit<1>)