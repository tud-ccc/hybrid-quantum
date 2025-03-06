module {
  func.func @main() -> (!qir.result, !qir.result) {
      %q1 = "qir.alloc"() : () -> (!qir.qubit)
      %q2 = "qir.alloc"() : () -> (!qir.qubit)

      %r1 = "qir.ralloc"() : () -> (!qir.result)
      %r2 = "qir.ralloc"() : () -> (!qir.result)

      "qir.measure"(%q1, %r1) : (!qir.qubit, !qir.result) -> ()
      "qir.measure"(%q2, %r2) : (!qir.qubit, !qir.result) -> ()
    return %r1,%r2 : () -> (!qir.result, !qir.result)
    }
}
