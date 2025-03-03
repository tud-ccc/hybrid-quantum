
  func.func @main() {
    %q0 = "qir.alloc" () : () -> (!qir.qubit)
    %r0 = "qir.ralloc" () : () -> (!qir.result)
    %const1 = arith.constant 0.34 : f64
    %const2 = arith.constant 0.735 : f64
    "qir.H" (%q0) : (!qir.qubit) -> ()
    "qir.Rz" (%q0, %const1) : (!qir.qubit, f64) -> ()
    "qir.Rz" (%q0, %const2) : (!qir.qubit, f64) -> ()
    "qir.measure" (%q0, %r0) : (!qir.qubit, !qir.result) -> ()
    return
  }
