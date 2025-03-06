module {
  func.func @main() -> (i1, i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 5 : index
    %step = arith.constant 1 : index
    %init = arith.constant 0 : i1
    %results1, %results2 = scf.for %i = %c0 to %c1 step %step iter_args(%r1 = %init, %r2 = %init) -> (i1, i1) {
      %q1 = "qir.alloc"() : () -> (!qir.qubit)
      %q2 = "qir.alloc"() : () -> (!qir.qubit)

      "qir.X" (%q1) : (!qir.qubit) -> ()

      %newR1 = "qir.ralloc"() : () -> (!qir.result)
      "qir.measure"(%q1, %newR1) : (!qir.qubit, !qir.result) -> ()
      %res0 = "qir.read_measurement" (%newR1) : (!qir.result) -> (i1)

      %newR2 = "qir.ralloc"() : () -> (!qir.result)
      "qir.measure"(%q2, %newR2) : (!qir.qubit, !qir.result) -> ()
      %res1 = "qir.read_measurement" (%newR2) : (!qir.result) -> (i1)
      scf.yield %res0, %res1 : i1, i1
    }
    
    return %results1, %results2 : i1, i1
  }
}
