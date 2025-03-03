module {
  func.func @main() -> (!qir.result, !qir.result) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 5 : index
    %step = arith.constant 1 : index
    %initResult = "qir.ralloc"() : () -> (!qir.result)
    %results1, %results2 = scf.for %i = %c0 to %c1 step %step iter_args(%r1 = %initResult, %r2 = %initResult) -> (!qir.result, !qir.result) {
    %q1 = "qir.alloc"() : () -> (!qir.qubit)
    %q2 = "qir.alloc"() : () -> (!qir.qubit)
    %newR1 = "qir.ralloc"() : () -> (!qir.result)
    %newR2 = "qir.ralloc"() : () -> (!qir.result)
    "qir.X"(%q1) : (!qir.qubit) -> ()
    "qir.measure"(%q1, %newR1) : (!qir.qubit, !qir.result) -> ()
    "qir.measure"(%q2, %newR2) : (!qir.qubit, !qir.result) -> ()
    scf.yield %newR1, %newR2 : !qir.result, !qir.result
}
    return %results1, %results2 : !qir.result, !qir.result
  }
}
